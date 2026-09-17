import core.config as config
from stt.speech import SpeechRecognizer
from stt.connection import Connection
from stt.sherpa import MicZipformerRecognizer
from stt.whisper import WhisperRecognizer

class STTManager:
    def __init__(self):
        self.speech = None
        self.sherpa = None
        self.whisper = None
        self.connection = Connection()
        self.is_mic = False

    def listen(self) -> str:
        try:
            if self.connection.is_online():
                if self.speech is None:
                    self.speech = SpeechRecognizer()
                return self.speech.listen()
            else:
                if config.STT_LANGUAGE == "vi":
                    if self.sherpa is None:
                        self.sherpa = MicZipformerRecognizer()
                    return self.sherpa.listen(silence_duration=1.5, volume_threshold=0.015)
                else:
                    if self.whisper is None:
                        self.whisper = WhisperRecognizer(model_size="small", language=config.STT_LANGUAGE, silence_duration=1.5, silence_threshold=0.015, start_timeout=None)
                    return self.whisper.listen()
        except Exception as e:
            print(f"[STT] {e}]")
            self.connection.stop()

