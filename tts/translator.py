import argostranslate.package
import argostranslate.translate
from core.app_context import AppContext

class Translator:
    def __init__(self, auto_setup=True, auto_warmup=True):
        self.src_lang = "vi"
        self.en_lang = "en"
        self.ja_lang = "ja"
        self.zh_lang = "zh"

        self.voice = AppContext.voice

        if auto_setup:
            self.setup_models()

        if auto_warmup:
            self.warm_up()

    def setup_models(self):
        print("[TRA] Checking language packs...")
        installed_languages = argostranslate.translate.get_installed_languages()
        installed_codes = [lang.code for lang in installed_languages]

        required_langs = [self.src_lang, self.en_lang, self.ja_lang, self.zh_lang]
        if all(lang in installed_codes for lang in required_langs):
            print("[TRA] The models are ready.")
            return

        print("[TRA] Model incomplete. Loading data from the server....")
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()

        # Add language pairs: vi -> en, en -> ja, en -> zh
        pairs = [
            (self.src_lang, self.en_lang),
            (self.en_lang, self.ja_lang),
            (self.en_lang, self.zh_lang),
        ]

        for src, tgt in pairs:
            try:
                pkg = next(p for p in available_packages if p.from_code == src and p.to_code == tgt)
                argostranslate.package.install_from_path(pkg.download())
                print(f" -> Package installed: {src} -> {tgt}")
            except StopIteration:
                print(f" -> Error: Package {src} -> {tgt} not found on the server.")
            except Exception as e:
                print(f" -> Failed to install {src} -> {tgt}: {e}")

    def warm_up(self):
        print("[TRA] Loading model into RAM (Warm-up)...")
        self.translate("Khởi động")
        print("[TRA] Fully loaded! Ready to translate at maximum speed.")

    def translate(self, text: str):
        if not text or str(text).isspace():
            return ""
        try:
            target_lang = self.voice.get_language()

            english_text = argostranslate.translate.translate(text, self.src_lang, self.en_lang)

            if target_lang == self.en_lang:
                return english_text

            if target_lang in [self.ja_lang, self.zh_lang]:
                return argostranslate.translate.translate(english_text, self.en_lang, target_lang)

            return english_text

        except Exception as e:
            return f"[Translation error]: {e}"
