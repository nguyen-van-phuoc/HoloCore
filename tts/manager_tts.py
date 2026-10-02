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
import sounddevice as sd
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal

from core import config
from tts.translator import Translator
from tts.manager_voice import VoiceManager
from live.live2d.render import Live2DWidget
import live.live2d.motion_mapping as mapping
from live.live2d.lip_sync_state import LipSyncState
from live.live2d.input_widget import TrueLiquidWidget

from korvatts import TTS

class TTSSignals(QObject):
    play_motion = pyqtSignal(str, int)


class TTSManager:
    def __init__(self,
                 voice: VoiceManager | None = None,
                 lip_sync: LipSyncState | None = None,
                 live_2d: Live2DWidget | None = None,
                 true_liquid_widget: TrueLiquidWidget | None = None):

        self.voice = voice
        self.lip_sync = lip_sync
        self.live_2d = live_2d
        self.true_liquid_widget = true_liquid_widget

        self.signals = TTSSignals()
        if self.live_2d:
            self.signals.play_motion.connect(self.live_2d.play_animation)
        if self.true_liquid_widget:
            self.true_liquid_widget.user_interrupt.connect(self.interrupt)

        self.session_id = 0
        self.session_lock = threading.Lock()

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.synth_thread = threading.Thread(target=self.synth_loop, daemon=True)
        self.player_thread = threading.Thread(target=self.player_loop, daemon=True)

        self.text_line = ""
        self.is_vietnamese = config.IS_VIETNAMESE

        self._init_tts_engine()

    def _init_tts_engine(self):
        print("[TTS] Initializing TTS Engine...")
        t0 = time.time()

        if self.is_vietnamese:
            print("[TTS] Mode: KorvaTTS (Tiếng Việt)")
            self.translator = None
            self.korvatts_engine = TTS()

            result = self.synthesize_korvatts("xin chào")
            if result:
                print(f"[TTS] KorvaTTS initialized and warmed up in {time.time() - t0:.2f}s")
            else:
                print(f"[TTS] KorvaTTS warmup failed!")

        else:
            print("[TTS] Mode: GPT-SoVITS")
            self.translator = Translator()
            self.sovits_url = config.SOVITS_URL

            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
            self.http_client = httpx.Client(timeout=60, limits=limits)

            if not self.is_port_available(9880):
                gpt_dir = Path(__file__).resolve().parent.parent / "tts" / "GPT-SoVITS"
                subprocess.Popen(f'start cmd /k ""{sys.executable}" api_v2.py"', cwd=str(gpt_dir), shell=True)

            while True:
                if self.is_port_available(9880):
                    print("[TTS] GPT-SoVITS server is online.")
                    break
                time.sleep(1)

            result = self.synthesize_sovits("言語")
            if result:
                print(f"[TTS] GPT-SoVITS initialized and warmed up in {time.time() - t0:.2f}s")
            else:
                print(f"[TTS] GPT-SoVITS warmup failed!")

    def is_port_available(self, port, host="127.0.0.1"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

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
        self.text_queue.put((text, self.session_id))

    def interrupt(self):
        with self.session_lock:
            self.session_id += 1
            current_sid = self.session_id
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                    self.text_queue.task_done()
                except queue.Empty:
                    break
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
        if hasattr(self, 'lip_sync') and self.lip_sync is not None:
            self.lip_sync.stop()
        print(f"[TTS] Interrupted! Session reset to #{current_sid}")

    def wait_until_done(self):
        self.text_queue.join()
        self.audio_queue.join()

    def synth_loop(self):
        while not self._stop_event.is_set():
            item = self.text_queue.get()
            if item is None:
                self.text_queue.task_done()
                self.audio_queue.put(None)
                break

            if isinstance(item, tuple):
                text_chunk, item_sid = item
            else:
                text_chunk, item_sid = item, self.session_id

            if item_sid != self.session_id:
                self.text_queue.task_done()
                continue

            self.text_line = text_chunk

            try:
                clean_chunk, motion_name = mapping.extract_motion_tag(text_chunk)
                if motion_name and item_sid == self.session_id:
                    self.play_animation_live_2d(motion_name)

                if clean_chunk:
                    if self.is_vietnamese:
                        text = clean_chunk
                        result = self.synthesize_korvatts(text)
                    else:
                        text = self.translator.translate(clean_chunk)
                        result = self.synthesize_sovits(text)

                    if result is not None and item_sid == self.session_id:
                        self.audio_queue.put((result, item_sid))

            except Exception as e:
                print(f"[TTS] Synth Loop Error: {e}")
            finally:
                self.text_queue.task_done()

    def player_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self.audio_queue.get()
            if item is None:
                self.audio_queue.task_done()
                break

            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], int):
                (audio, sr), item_sid = item[0], item[1]
            else:
                audio, sr = item
                item_sid = self.session_id

            if item_sid != self.session_id:
                self.audio_queue.task_done()
                continue

            try:
                silence = np.zeros(int(sr * config.AUDIO_TAIL_PADDING_MS / 1000), dtype=np.float32)
                full_audio = np.concatenate([audio, silence])

                if self.lip_sync is not None:
                    self.lip_sync.start(full_audio, sr)

                if self.lip_sync is not None:
                    self.lip_sync.start(full_audio, sr)

                self._play_interruptible(full_audio, sr, item_sid)

                if self.lip_sync is not None:
                    self.lip_sync.stop()

            except Exception as e:
                print(f"[TTS] Player Loop Error: {e}")
            finally:
                self.audio_queue.task_done()

    def _play_interruptible(self, audio, sr, sid):
        done = threading.Event()
        pos = 0

        def callback(outdata, frames, time_info, status):
            nonlocal pos
            if sid != self.session_id or self._stop_event.is_set():
                outdata.fill(0)
                raise sd.CallbackAbort
            chunk = audio[pos:pos + frames]
            n = len(chunk)
            outdata[:n, 0] = chunk
            if n < frames:
                outdata[n:] = 0
                raise sd.CallbackStop
            pos += n

        with sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                             latency="low", callback=callback,
                             finished_callback=done.set):
            done.wait(timeout=len(audio) / sr + 2)

    def synthesize_korvatts(self, text: str):
        if not text.strip():
            return None

        try:
            audio_data, duration = self.korvatts_engine.synthesize(
                text=text,
                voice="ngoc_huyen",
                lang="vi",
                total_steps=32,
                speed=1.65,
            )

            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            return audio_data, 44100
        except Exception as e:
            print(f"\n[TTS ERROR] KorvaTTS Synthesis: {e}")
            return None

    def synthesize_sovits(self, text: str):
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
            print(f"\n[TTS ERROR] SoVITS HTTP {e.response.status_code}: {e.response.text[:200]}")
            return None
        except Exception as e:
            print(f"\n[TTS ERROR] SoVITS Synthesis: {e}")
            return None

    def play_animation_live_2d(self, motion_name: str):
        if not self.live_2d:
            return

        try:
            if motion_name not in mapping.MOTION_MAPPING:
                print(f"[TTS ANIMATION] Unknown motion: {motion_name}")
                return
            group, indexs = mapping.MOTION_MAPPING[motion_name]
            index = random.choice(indexs)
            print(f"\n[INFO] Motion: {motion_name} | Group: {group} | Index: {index}")
            self.signals.play_motion.emit(group, index)
        except Exception as e:
            print(f"[TTS ANIMATION ERROR] {e}")