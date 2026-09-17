import io
import sys
import time
import httpx
import queue
import random
import socket
import threading
import subprocess
import numpy as np
import soundfile as sf
from core import config
from pathlib import Path
import sounddevice as sd
from tts.translator import Translator
from tts.manager_voice import VoiceManager
from live.live2d.render import Live2DWidget
import live.live2d.motion_mapping as mapping
from PyQt6.QtCore import QObject, pyqtSignal
from live.live2d.lip_sync_state import LipSyncState
from live.live2d.input_widget import TrueLiquidWidget

class TTSSignals(QObject):
    play_motion = pyqtSignal(str, int)

class GPTSoVits:
    def __init__(self, voice: VoiceManager | None = None,
                 lip_sync: LipSyncState | None = None,
                 live_2d: Live2DWidget | None = None,
                 true_liquid_widget: TrueLiquidWidget | None = None
    ):

        self.sovits_url = "http://127.0.0.1:9880/tts"
        self.voice = voice

        self.translator = Translator()

        self.samplerate = 24000

        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        self.http_client = httpx.Client(timeout=60, limits=limits)

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self._stop_event = threading.Event()
        self.synth_thread = threading.Thread(target=self.synth_loop, daemon=True)
        self.player_thread = threading.Thread(target=self.player_loop, daemon=True)

        self.lip_sync = lip_sync
        self.live_2d = live_2d
        self.signals = TTSSignals()
        self.signals.play_motion.connect(self.live_2d.play_animation)

        self.text_line = ""

        self.true_liquid_widget = true_liquid_widget

        self.init()

    def is_port_available(self, port, host="127.0.0.1"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    def init(self):
        print("[TTS] Initializing GPT-SoViTs...")

        if not self.is_port_available(9880):
            gpt_dir = Path(__file__).resolve().parent.parent / "tts" / "GPT-SoVITS"
            subprocess.Popen(f'start cmd /k ""{sys.executable}" api_v2.py"', cwd=str(gpt_dir), shell=True,)

        while True:
            if self.is_port_available(9880):
                print("[TTS] GPT-SoViTs initialized")
                self.warmup_gpt_sovits()
                break

    def warmup_gpt_sovits(self):
        print("[TTS] Warming up GPT-SoViTs...")
        t0 = time.time()
        result = self.synthesize_sovits("言語")
        if result is not None:
            print(f"[TTS] Warmup completed in {time.time() - t0:.2f}s")
        else:
            print(f"[TTS] Warmup failed after {time.time() - t0:.2f}s")

    def start(self):
        self.synth_thread.start()
        self.player_thread.start()

    def stop(self):
        self._stop_event.set()
        self.text_queue.put(None)

    def wait(self):
        self.synth_thread.join()
        self.player_thread.join()

    def speak(self, text):
        if not text.strip():
            return
        self.text_queue.put(text)

    def wait_until_done(self):
        self.text_queue.join()
        self.audio_queue.join()

    def translate(self, text):
        return self.translator.translate(text)

    def synth_loop(self):
        while not self._stop_event.is_set():
            text_chunk = self.text_queue.get()
            self.text_line = text_chunk
            if text_chunk is None:
                self.text_queue.task_done()
                self.audio_queue.put(None)
                break
            try:
                clean_chunk, motion_name = mapping.extract_motion_tag(text_chunk)
                if motion_name:
                    self.play_animation_live_2d(motion_name)
                text = self.translate(clean_chunk) if clean_chunk else None
                result = self.synthesize_sovits(text) if text else None
                if result is not None:
                    self.audio_queue.put(result)
            except Exception as e:
                print(f"[TTS] Error {e}")
            finally:
                self.text_queue.task_done()

    def player_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self.audio_queue.get()
            if item is None:
                self.audio_queue.task_done()
                break
            audio, sr = item
            try:
                silence = np.zeros(int(sr * config.AUDIO_TAIL_PADDING_MS / 1000), dtype=np.float32)
                full_audio = np.concatenate([audio, silence])
                if self.lip_sync is not None:
                    self.lip_sync.start(full_audio, sr)

                sd.play(np.concatenate([audio, silence]), samplerate=sr)
                sd.wait()
                if self.lip_sync is not None:
                    self.lip_sync.stop()
            except Exception as e:
                print(f"[TTS] Error {e}")
            finally:
                self.audio_queue.task_done()

    def synthesize_sovits(self, text):
        if not text.strip():
            return None

        body = {
            "text": text,
            "text_lang": self.voice.get_language(),
            "ref_audio_path": self.voice.get_path_voice(),
            "prompt_text": self.voice.get_reference_text(),
            "prompt_lang": self.voice.get_language(),
            "media_type": "wav",
            "streaming_mode": False,
        }

        try:
            resp = self.http_client.post(self.sovits_url, json=body)
            resp.raise_for_status()
            audio_data, samplerate = sf.read(io.BytesIO(resp.content), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            return audio_data, samplerate

        except httpx.HTTPStatusError as e:
            print(f"\n[TTS ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}")
            return None
        except Exception as e:
            print(f"\n[TTS ERROR] {e}")
            return None

    def play_animation_live_2d(self, motion_name: str):
        try:
            if motion_name not in mapping.MOTION_MAPPING:
                print(f"[TTS ANIMATION] Unknown motion: {motion_name}")
                return
            group, indexs = mapping.MOTION_MAPPING[motion_name]
            index = random.choice(indexs)
            print(f"\n[INFO] Motion: {motion_name} | Group: {group} | Index: {index}")
            self.signals.play_motion.emit(group, index)
        except Exception as e:
            print(f"[TTS ANIMATION] {e}")
