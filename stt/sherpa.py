import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import queue
import logging
import collections
import numpy as np
import sherpa_onnx
import sounddevice as sd
import sentencepiece as spm
from huggingface_hub import snapshot_download

class MicZipformerRecognizer:
    def __init__(self, repo_id="hynt/Zipformer-30M-RNNT-6000h", provider="cpu"):
        logging.getLogger("sherpa_onnx").setLevel(logging.INFO)
        print("[ZIP] Loading/checking Zipformer 30M model...")

        self.model_dir = snapshot_download(repo_id=repo_id)

        encoder_path = os.path.join(self.model_dir, "encoder-epoch-20-avg-10.int8.onnx")
        decoder_path = os.path.join(self.model_dir, "decoder-epoch-20-avg-10.onnx")
        joiner_path = os.path.join(self.model_dir, "joiner-epoch-20-avg-10.int8.onnx")
        bpe_path = os.path.join(self.model_dir, "bpe.model")

        tokens_path = os.path.join(self.model_dir, "tokens.txt")
        if not os.path.exists(tokens_path):
            sp = spm.SentencePieceProcessor()
            sp.load(bpe_path)
            with open(tokens_path, "w", encoding="utf-8") as f:
                for i in range(sp.get_piece_size()):
                    f.write(f"{sp.id_to_piece(i)} {i}\n")

        self.sample_rate = 16000

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=2,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
        )
        print("[ZIP] Model initialization successful!")

    def listen(self, silence_duration=2.0, volume_threshold=0.015) -> str:
        q = queue.Queue()
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(indata.copy())

        chunk_duration = 0.1
        blocksize = int(self.sample_rate * chunk_duration)
        max_silence_chunks = int(silence_duration / chunk_duration)

        pre_speech_chunks = collections.deque(maxlen=5)
        audio_buffer = []

        has_started_speaking = False
        silence_chunks_count = 0

        print("[STT] Waiting for you to speak... (Will automatically cut off after 2 seconds of silence)")
        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                            dtype='float32', blocksize=blocksize, callback=audio_callback):
            while True:
                chunk = q.get()
                rms_volume = np.sqrt(np.mean(chunk ** 2))
                is_speaking = rms_volume > volume_threshold
                if not has_started_speaking:
                    if is_speaking:
                        has_started_speaking = True
                        print("  -> Recording...")
                        audio_buffer.extend(list(pre_speech_chunks))
                        audio_buffer.append(chunk)
                        silence_chunks_count = 0
                    else:
                        pre_speech_chunks.append(chunk)
                else:
                    audio_buffer.append(chunk)
                    if not is_speaking:
                        silence_chunks_count += 1
                        if silence_chunks_count >= max_silence_chunks:
                            print(f"  -> Silence for {silence_duration}s, starting text processing...")
                            break
                    else:
                        silence_chunks_count = 0
        final_audio = np.concatenate(audio_buffer).flatten()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, final_audio)
        self.recognizer.decode_stream(stream)
        result_text = stream.result.text.lower().capitalize()
        return result_text

