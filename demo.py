import os  
import sys  
import threading  
import numpy as np  
import sounddevice as sd  
from datetime import timedelta  
import webrtcvad  
import json  
from vosk import Model, KaldiRecognizer  
import queue  
import time  
import logging  
  
import torch  
from speechbrain.pretrained import EncoderClassifier  
  
# Import FAISS for efficient similarity search  
import faiss  
  
"""  
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.  
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.   
That is dangerous, since it can degrade performance or cause incorrect results.   
The best thing to do is to ensure that only a single OpenMP runtime is linked into the process,   
e.g. by avoiding static linking of the OpenMP runtime in any library.   
As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE   
to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results.  
For more information, please see http://openmp.llvm.org/  
"""  
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  
  
# Set up logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
class SpeakerIdentifier:  
    """  
    A class to perform real-time speaker identification and transcription using FAISS for efficient vector search.  
    """  
  
    # Constants for easy tweaking  
    MODEL_NAME = "models/vosk-model-small-en-us-0.15"  # Vosk model directory  
    CHANNELS = 1                                      # Number of audio channels (mono)  
    FRAME_DURATION_MS = 30                            # Frame size in milliseconds  
    VAD_AGGRESSIVENESS = 2                            # VAD aggressiveness (0-3)  
    ME_THRESHOLD = 0.7                                # Similarity threshold for 'Me' classification  
    NOT_ME_THRESHOLD = 0.6                            # Similarity threshold for 'Not Me' speaker identification  
    MIN_SEGMENT_DURATION = 1.0                        # Minimum duration of a segment in seconds  
  
    def __init__(self):  
        # Adjust sample rate based on the default input device  
        try:  
            default_input_device = sd.query_devices(kind='input')  
            default_sample_rate = int(default_input_device['default_samplerate'])  
            self.sample_rate = default_sample_rate if default_sample_rate else 16000  
        except Exception as e:  
            logger.error(f"Could not determine default sample rate: {e}")  
            self.sample_rate = 16000  
  
        if self.sample_rate != 16000:  
            logger.warning(f"Default sample rate is {self.sample_rate} Hz. Adjusting to match the microphone's capabilities.")  
  
        # Initialize Vosk model  
        if not os.path.exists(self.MODEL_NAME):  
            logger.error(f"Vosk model '{self.MODEL_NAME}' not found. Please download and place it in the script directory.")  
            sys.exit(1)  
        logger.info("Loading Vosk model...")  
        self.model = Model(self.MODEL_NAME)  
        logger.info("Vosk model loaded.")  
  
        # Initialize the voice encoder using SpeechBrain's ECAPA-TDNN model  
        logger.info("Initializing voice encoder...")  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.encoder = EncoderClassifier.from_hparams(  
            source="speechbrain/spkrec-ecapa-voxceleb",  
            savedir="models/spkrec-ecapa-voxceleb",  
            run_opts={"device": self.device},  
        )  
        logger.info("Voice encoder initialized.")  
  
        # Determine the embedding dimension  
        self.embedding_dim = self.get_embedding_dim()  
        logger.info(f"Embedding dimension: {self.embedding_dim}")  
  
        # Initialize FAISS index for 'Me' embeddings collected during enrollment  
        self.me_index = faiss.IndexFlatIP(self.embedding_dim)  
        self.me_embeddings = []  
        self.me_labels = []  
  
        # Initialize FAISS index for 'Not Me' embeddings  
        self.not_me_index = faiss.IndexFlatIP(self.embedding_dim)  
        self.not_me_id_to_label = {}  
        self.next_speaker_id = 1  # For 'Not Me' speakers  
  
        # Initialize WebRTC VAD  
        self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)  
  
        # Buffer to hold audio frames  
        self.audio_queue = queue.Queue()  
        self.running = True  
  
        # For timestamp calculations  
        self.recording_start_time = None  
  
        # List to store transcribed segments  
        self.transcript_segments = []  
  
        # Thread for VAD collector  
        self.vad_thread = None  
  
    def get_embedding_dim(self):  
        """  
        Determines the embedding dimension by running a dummy input through the encoder.  
        """  
        # Create a dummy audio tensor  
        dummy_audio = torch.zeros((1, 16000)).to(self.device)  # 1 second of silence at 16kHz  
        with torch.no_grad():  
            embeddings = self.encoder.encode_batch(dummy_audio)  
        embedding_dim = embeddings.shape[-1]  
        return embedding_dim  
  
    def record_enrollment(self, num_samples=3, duration=5):  
        """  
        Records multiple audio samples for enrollment and stores 'Me' embeddings in FAISS index.  
        """  
        for i in range(num_samples):  
            logger.info(f"Recording enrollment sample {i + 1}/{num_samples} for {duration} seconds...")  
            try:  
                audio_data = sd.rec(int(duration * self.sample_rate),  
                                    samplerate=self.sample_rate,  
                                    channels=self.CHANNELS,  
                                    dtype='int16')  
                sd.wait()  
                audio_data = np.squeeze(audio_data)  
                # Extract voiced audio using VAD  
                voiced_audio = self.extract_voiced_audio(audio_data)  
                # Check if voiced audio is sufficient  
                if len(voiced_audio) < self.sample_rate * self.MIN_SEGMENT_DURATION:  
                    logger.warning("Voiced audio segment is too short. Try speaking clearly into the microphone.")  
                    continue  
                # Create embedding from voiced audio  
                embedding = self.create_embedding(voiced_audio)  
                if embedding is not None:  
                    # Add the embedding to the list  
                    self.me_embeddings.append(embedding)  
                    # Add embedding to 'Me' FAISS index  
                    self.add_embedding_to_me_index(embedding)  
            except Exception as e:  
                logger.error(f"Error during enrollment recording: {e}")  
                sys.exit(1)  
        if self.me_embeddings:  
            logger.info(f"{len(self.me_embeddings)} enrollment embeddings created and added to 'Me' FAISS index.")  
        else:  
            logger.error("No embeddings were created during enrollment.")  
            sys.exit(1)  
  
    def extract_voiced_audio(self, audio_data):  
        """  
        Extracts voiced frames from the audio data using VAD.  
        """  
        frame_size = int(self.FRAME_DURATION_MS / 1000 * self.sample_rate)  # Frame size in samples  
        frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]  
        voiced_frames = []  
        for frame in frames:  
            if len(frame) < frame_size:  
                # Pad the frame if it's shorter than the expected frame size  
                frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')  
            frame_bytes = frame.tobytes()  
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)  
            if is_speech:  
                voiced_frames.extend(frame)  
        voiced_audio = np.array(voiced_frames, dtype=np.int16)  
        return voiced_audio  
  
    def create_embedding(self, audio_data):  
        """  
        Creates a voice embedding from audio data using SpeechBrain's encoder.  
        """  
        # Check if the audio data is long enough  
        if len(audio_data) < self.sample_rate * self.MIN_SEGMENT_DURATION:  
            logger.debug("Audio segment is too short for embedding.")  
            return None  
  
        # Convert audio data to float32 numpy array and normalize  
        audio_float32 = audio_data.astype(np.float32) / 32768.0  
  
        # Convert numpy array to PyTorch tensor and add batch dimension  
        audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(self.device)  
  
        # Generate the embedding  
        with torch.no_grad():  
            embeddings = self.encoder.encode_batch(audio_tensor)  
        embedding = embeddings.squeeze().cpu().numpy()  
  
        # Normalize the embedding  
        embedding = embedding / np.linalg.norm(embedding)  
        return embedding  
  
    def add_embedding_to_me_index(self, embedding):  
        """  
        Adds a 'Me' embedding to the 'Me' FAISS index.  
        """  
        # Reshape embedding to 2D array for FAISS  
        embedding = embedding.reshape(1, -1)  
        # Add to 'Me' FAISS index  
        self.me_index.add(embedding)  
        logger.info(f"Added new 'Me' embedding to FAISS index. Total 'Me' embeddings: {self.me_index.ntotal}")  
  
    def add_embedding_to_not_me_index(self, embedding, label):  
        """  
        Adds a 'Not Me' embedding to the 'Not Me' FAISS index and updates the id_to_label mapping.  
        """  
        # Reshape embedding to 2D array for FAISS  
        embedding = embedding.reshape(1, -1)  
        # Add to 'Not Me' FAISS index  
        self.not_me_index.add(embedding)  
        # The index ID of the added embedding  
        index_id = self.not_me_index.ntotal - 1  
        # Update id_to_label mapping  
        self.not_me_id_to_label[index_id] = label  
        logger.info(f"Added new embedding to 'Not Me' FAISS index with label '{label}'. Total 'Not Me' embeddings: {self.not_me_index.ntotal}")  
  
    def vad_collector(self):  
        """  
        Collects voiced frames from the audio queue using VAD.  
        """  
        frame_duration = self.FRAME_DURATION_MS / 1000.0  
        num_padding_frames = int(0.5 / frame_duration)  # 0.5 seconds of padding  
        ring_buffer = []  
        triggered = False  
  
        voiced_frames = []  
        start_time = None  # Initialize start_time  
  
        while self.running or not self.audio_queue.empty():  
            try:  
                frame = self.audio_queue.get(timeout=1)  
            except queue.Empty:  
                if not self.running and self.audio_queue.empty():  
                    break  
                else:  
                    continue  
  
            is_speech = self.vad.is_speech(frame, self.sample_rate)  
            current_time = time.time()  
  
            if not triggered:  
                ring_buffer.append((frame, is_speech, current_time))  
                num_voiced = len([f for f, speech, _ in ring_buffer if speech])  
                if num_voiced > 0.9 * len(ring_buffer):  
                    triggered = True  
                    voiced_frames.extend([f for f, _, _ in ring_buffer])  
                    start_time = ring_buffer[0][2]  
                    ring_buffer = []  
                elif len(ring_buffer) > num_padding_frames:  
                    ring_buffer.pop(0)  
            else:  
                voiced_frames.append(frame)  
                ring_buffer.append((frame, is_speech, current_time))  
                num_unvoiced = len([f for f, speech, _ in ring_buffer if not speech])  
                if num_unvoiced > 0.9 * len(ring_buffer):  
                    end_time = ring_buffer[-1][2]  
                    triggered = False  
                    self.process_segment(voiced_frames, start_time, end_time)  
                    start_time = None  # Reset start_time  
                    ring_buffer = []  
                    voiced_frames = []  
                elif len(ring_buffer) > num_padding_frames:  
                    ring_buffer.pop(0)  
  
        # Process any remaining frames regardless of triggered state  
        if voiced_frames and start_time is not None:  
            end_time = time.time()  
            self.process_segment(voiced_frames, start_time, end_time)  
        else:  
            logger.debug("No valid segment to process on exit.")  
  
    def process_segment(self, frames, start_time, end_time):  
        """  
        Processes a voiced segment: speaker identification and transcription.  
        """  
        # Initialize recording start time if not set  
        if self.recording_start_time is None:  
            self.recording_start_time = start_time  
  
        # Convert frames to numpy array  
        segment = b''.join(frames)  
        segment_audio = np.frombuffer(segment, dtype=np.int16)  
  
        # Ignore short segments  
        duration = end_time - start_time if start_time else 0  
        audio_duration = len(segment_audio) / self.sample_rate  
        if duration < self.MIN_SEGMENT_DURATION or audio_duration < self.MIN_SEGMENT_DURATION:  
            logger.debug("Segment duration is too short, skipping.")  
            return  
  
        # Transcription  
        transcript = self.transcribe_audio(segment_audio).strip()  
  
        # Ignore segments with empty transcripts  
        if not transcript:  
            logger.debug("Transcription is empty, skipping segment.")  
            return  
  
        # Speaker identification  
        try:  
            # Create an embedding for the current segment  
            segment_embedding = self.create_embedding(segment_audio)  
            if segment_embedding is None:  
                logger.debug("Embedding not created due to short audio segment.")  
                return  
  
            # Normalize embedding  
            segment_embedding = segment_embedding / np.linalg.norm(segment_embedding)  
  
            # Compare to 'Me' embeddings collected during enrollment  
            me_similarity = self.compare_to_me(segment_embedding)  
  
            # Log the similarity score  
            logger.info(f"Similarity to 'Me': {me_similarity:.4f}")  
  
            if me_similarity >= self.ME_THRESHOLD:  
                speaker_label = "Me"  
                # Do not add new 'Me' embeddings during runtime  
            else:  
                # Not 'Me' speaker  
                speaker_label = self.identify_not_me_speaker(segment_embedding)  
        except Exception as e:  
            logger.error(f"Error during speaker identification: {e}")  
            speaker_label = "Unknown"  
  
        # Display the result  
        relative_start_time = start_time - self.recording_start_time  
        relative_end_time = end_time - self.recording_start_time  
        start_td = str(timedelta(seconds=int(relative_start_time)))  
        end_td = str(timedelta(seconds=int(relative_end_time)))  
        logger.info(f"[{start_td} - {end_td}] [{speaker_label}] {transcript}")  
  
        # Collect the segment for the final transcript  
        self.transcript_segments.append({  
            'start_time': relative_start_time,  
            'end_time': relative_end_time,  
            'speaker_label': speaker_label,  
            'transcript': transcript  
        })  
  
    def compare_to_me(self, segment_embedding):  
        """  
        Computes the maximum similarity between the segment embedding and 'Me' embeddings.  
        """  
        if self.me_index.ntotal == 0:  
            return 0.0  # No 'Me' embeddings to compare  
        # Reshape embedding for FAISS  
        segment_embedding = segment_embedding.reshape(1, -1)  
        # Search 'Me' index  
        D, _ = self.me_index.search(segment_embedding, 1)  
        similarity = D[0][0]  
        return similarity  
  
    def identify_not_me_speaker(self, segment_embedding):  
        """  
        Identifies the speaker among 'Not Me' speakers or assigns a new speaker label.  
        """  
        # Reshape embedding for FAISS  
        segment_embedding = segment_embedding.reshape(1, -1)  
  
        # Avoid adding multiple embeddings for the same speaker unless confident  
        if self.not_me_index.ntotal == 0:  
            # No 'Not Me' speakers yet  
            speaker_label = f"Speaker {self.next_speaker_id}"  
            self.add_embedding_to_not_me_index(segment_embedding, speaker_label)  
            logger.info(f"Assigned new speaker label: {speaker_label}")  
            self.next_speaker_id += 1  
            return speaker_label  
  
        # Search 'Not Me' index  
        D, I = self.not_me_index.search(segment_embedding, 1)  
        similarity = D[0][0]  
        index_id = I[0][0]  
  
        # Log the similarity score  
        logger.info(f"Similarity to closest 'Not Me' speaker: {similarity:.4f}")  
  
        if similarity >= self.NOT_ME_THRESHOLD:  
            # Assign the existing speaker label  
            speaker_label = self.not_me_id_to_label[index_id]  
            # Optionally, do not add new embeddings to prevent vector space pollution  
        else:  
            # New 'Not Me' speaker  
            speaker_label = f"Speaker {self.next_speaker_id}"  
            self.add_embedding_to_not_me_index(segment_embedding, speaker_label)  
            logger.info(f"Assigned new speaker label: {speaker_label}")  
            self.next_speaker_id += 1  
  
        return speaker_label  
  
    def transcribe_audio(self, audio_data):  
        """  
        Transcribes audio data using Vosk.  
        """  
        rec = KaldiRecognizer(self.model, self.sample_rate)  
        rec.SetWords(True)  
  
        audio_bytes = audio_data.tobytes()  
        rec.AcceptWaveform(audio_bytes)  
        res = rec.Result()  
        res_json = json.loads(res)  
        transcript = res_json.get('text', '')  
        return transcript  
  
    def audio_callback(self, indata, frames, time_info, status):  
        """  
        Callback function for real-time audio processing.  
        """  
        if status:  
            logger.warning(f"Audio status: {status}")  
  
        # Convert audio input to bytes and put into the queue  
        self.audio_queue.put(indata.tobytes())  
  
    def start_listening(self):  
        """  
        Starts the real-time audio stream and voice activity detection.  
        """  
        self.running = True  
        logger.info("Starting real-time processing. Press Ctrl+C to stop.")  
  
        # Start VAD collector in a separate thread  
        self.vad_thread = threading.Thread(target=self.vad_collector)  
        self.vad_thread.daemon = True  
        self.vad_thread.start()  
  
        # Start audio stream  
        try:  
            with sd.InputStream(  
                samplerate=self.sample_rate,  
                channels=self.CHANNELS,  
                dtype='int16',  
                blocksize=int(self.sample_rate * (self.FRAME_DURATION_MS / 1000.0)),  
                callback=self.audio_callback  
            ):  
                while self.running:  
                    time.sleep(0.1)  
        except Exception as e:  
            logger.error(f"Error during audio streaming: {e}")  
            self.running = False  
  
        # Wait for the audio queue to empty  
        while not self.audio_queue.empty():  
            time.sleep(0.1)  
        self.vad_thread.join()  
  
    def wait_for_completion(self):  
        """  
        Waits for the VAD collector thread to finish.  
        """  
        if self.vad_thread and self.vad_thread.is_alive():  
            self.vad_thread.join()  
  
    def print_final_transcript(self):  
        """  
        Prints the final transcript collected during the session.  
        """  
        logger.info("\nFinal Transcript:\n")  
        for segment in self.transcript_segments:  
            if segment['transcript'].strip():  # Only print if transcript is not empty  
                start_td = str(timedelta(seconds=int(segment['start_time'])))  
                end_td = str(timedelta(seconds=int(segment['end_time'])))  
                print(f"[{start_td} - {end_td}] [{segment['speaker_label']}] {segment['transcript']}")  
  
def main():  
    speaker_id = SpeakerIdentifier()  
    try:  
        speaker_id.record_enrollment(num_samples=3, duration=5)  
        speaker_id.start_listening()  
    except KeyboardInterrupt:  
        logger.info("Interrupted by user.")  
        speaker_id.running = False  # Ensure all threads exit  
    except Exception as e:  
        logger.error(f"An error occurred: {e}")  
    finally:  
        speaker_id.wait_for_completion()  
        speaker_id.print_final_transcript()  
  
if __name__ == "__main__":  
    main()  


"""
INFO:__main__:Voice encoder initialized.
INFO:__main__:Embedding dimension: 192
INFO:__main__:Recording enrollment sample 1/3 for 5 seconds...
INFO:__main__:Added new 'Me' embedding to FAISS index. Total 'Me' embeddings: 1
INFO:__main__:Recording enrollment sample 2/3 for 5 seconds...
INFO:__main__:Added new 'Me' embedding to FAISS index. Total 'Me' embeddings: 2
INFO:__main__:Recording enrollment sample 3/3 for 5 seconds...
INFO:__main__:Added new 'Me' embedding to FAISS index. Total 'Me' embeddings: 3
INFO:__main__:3 enrollment embeddings created and added to 'Me' FAISS index.
INFO:__main__:Starting real-time processing. Press Ctrl+C to stop.
INFO:__main__:Similarity to 'Me': 0.8749
INFO:__main__:[0:00:00 - 0:00:07] [Me] hi my name is fabian can you keep track of this voice for have more than five minutes
INFO:__main__:Similarity to 'Me': 0.4935
INFO:__main__:Added new embedding to 'Not Me' FAISS index with label 'Speaker 1'. Total 'Not Me' embeddings: 1
INFO:__main__:Assigned new speaker label: Speaker 1
INFO:__main__:[0:00:24 - 0:00:29] [Speaker 1] eight ten answers to questions
INFO:__main__:Similarity to 'Me': 0.3721
INFO:__main__:Similarity to closest 'Not Me' speaker: 0.7559
INFO:__main__:[0:00:29 - 0:00:33] [Speaker 1] then we wrapped us all into a dataset
INFO:__main__:Similarity to 'Me': 0.2664
INFO:__main__:Similarity to closest 'Not Me' speaker: 0.6841
INFO:__main__:[0:00:33 - 0:00:35] [Speaker 1] the we have the dataset
INFO:__main__:Similarity to 'Me': 0.3929
INFO:__main__:Similarity to closest 'Not Me' speaker: 0.7553
INFO:__main__:[0:00:35 - 0:00:39] [Speaker 1] you're able to start thinking about using rigas
INFO:__main__:Similarity to 'Me': 0.3921
INFO:__main__:Similarity to closest 'Not Me' speaker: 0.7829
INFO:__main__:[0:00:40 - 0:00:58] [Speaker 1] so the way that we're going to do this is by having this create rag as dataset option because remember our answers that we have are considered the ground truth answers but we need to evaluate our l lens answers to these questions right so we have a god truth truth truth through
INFO:__main__:Similarity to 'Me': 0.7118
INFO:__main__:[0:01:01 - 0:01:06] [Me] what about this do you know who's speaking now based on my profile
INFO:__main__:Similarity to 'Me': 0.8236
INFO:__main__:[0:01:14 - 0:01:20] [Me] pretty cool that you can keep track now without having to pollute that factors space
INFO:__main__:Similarity to 'Me': 0.4327
INFO:__main__:Similarity to closest 'Not Me' speaker: 0.7026
INFO:__main__:[0:01:29 - 0:01:36] [Speaker 1] helper function that we are built above ever gonna pass in this specific retriever and that's all we have to do
^CINFO:__main__:Interrupted by user.
INFO:__main__:Similarity to 'Me': 0.3581
INFO:__main__:Similarity to closest 'Not Me' speaker: 0.6591
INFO:__main__:[0:01:37 - 0:01:42] [Speaker 1] we could ask a question what is rag and we rag says virtual
INFO:__main__:
Final Transcript:

[0:00:00 - 0:00:07] [Me] hi my name is fabian can you keep track of this voice for have more than five minutes
[0:00:24 - 0:00:29] [Speaker 1] eight ten answers to questions
[0:00:29 - 0:00:33] [Speaker 1] then we wrapped us all into a dataset
[0:00:33 - 0:00:35] [Speaker 1] the we have the dataset
[0:00:35 - 0:00:39] [Speaker 1] you're able to start thinking about using rigas
[0:00:40 - 0:00:58] [Speaker 1] so the way that we're going to do this is by having this create rag as dataset option because remember our answers that we have are considered the ground truth answers but we need to evaluate our l lens answers to these questions right so we have a god truth truth truth through
[0:01:01 - 0:01:06] [Me] what about this do you know who's speaking now based on my profile
[0:01:14 - 0:01:20] [Me] pretty cool that you can keep track now without having to pollute that factors space
[0:01:29 - 0:01:36] [Speaker 1] helper function that we are built above ever gonna pass in this specific retriever and that's all we have to do
[0:01:37 - 0:01:42] [Speaker 1] we could ask a question what is rag and we rag says virtual
"""
