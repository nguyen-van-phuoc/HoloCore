from pathlib import Path

class VoiceManager:
    def __init__(self):
        self.dir_base = Path.cwd()
        self.data_voice = {
            "holo": {
                "path_voice": str(self.dir_base.joinpath("audio", "holo.mp3")),
                "reference_text": "そうか、無視は村の人間ではなかったんじゃない?うんすまに脳を忘れたったわ",
                "language": "ja"
            },

            "mahiru": {
                "path_voice": str(self.dir_base.joinpath("audio", "mahiru.mp3")),
                "reference_text": "大家好我是呆毛，所有的考試只有一隻費，從來沒有愛著任何人的。但是昨天，是有一個人的，他給我扣了一頂假推的帽子",
                "language": "zh"
            },

            "fuwa": {
                "path_voice": str(self.dir_base.joinpath("audio", "fuwa.mp3")),
                "reference_text": "Step two: You just gotta keep practicing because practices make perfect. Let's try to pronounce this words.",
                "language": "en"
            }
        }
        self.current_voice = "holo"

    def get_path_voice(self):
        return self.data_voice[self.current_voice]["path_voice"]

    def get_reference_text(self):
        return self.data_voice[self.current_voice]["reference_text"]

    def get_language(self):
        return self.data_voice[self.current_voice]["language"]

    def set_current_voice(self, name):
        if name in self.data_voice:
            self.current_voice = name
        else:
            print("[VOI] Voice not found")

    def set_voice_character(self, character: str):
        if character == "COMMAND:HOLO_VOICE":
            self.set_current_voice("holo")
        elif character == "COMMAND:MAHIRU_VOICE":
            self.set_current_voice("mahiru")
        elif character == "COMMAND:FUWA_VOICE":
            self.set_current_voice("fuwa")
        else:
            print("[VOI] Voice not found")







