import core.config as config
import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self):
        self.recognition = sr.Recognizer()
        self.recognition.pause_threshold = 1.5
        self.source = sr.Microphone()
        self.ready = False

    def ensure_ready(self) -> None:
        if not self.ready:
            with self.source:
                self.recognition.adjust_for_ambient_noise(self.source, duration=1)

    def listen(self):
        self.ensure_ready()
        try:
            with self.source:
                audio = self.recognition.listen(self.source)
            return self.recognition.recognize_google(audio, language=config.STT_LANGUAGE)
        except sr.UnknownValueError:
            print("[STT] Can't hear clearly!")
        except sr.RequestError as e:
            print("[STT] Google STT API error: %s", e)
        except KeyboardInterrupt:
            pass
        return None