import cv2
from core import config
from io import BytesIO
from ollama import Client
from PIL import Image, ImageGrab

class ImageAnalyzer:
    def __init__(self):
        self.client = Client(host=config.OLLAMA_BASE_URL)
        self.system_prompt = """
        You are an OCR and screen understanding assistant.
        Your task is to answer the user's question using only the provided images.
        Rules:
        - Read all visible text accurately.
        - Describe only what is relevant to the question.
        - Do not guess missing or obscured content.
        - Do not use external knowledge.
        - If the answer is not visible, reply exactly: "Unknown".
        - Keep the response concise.
        """

    def analysis_camera(self, device, question):
        print("[VIS] Receiving images...")
        image = self.image_monitor() if device == "monitor" else self.image_camera()
        print("[VIS] Analyzing images...")
        return self._run_vision(image, question)

    def _run_vision(self, image, question):
        buf_img = BytesIO()
        image.save(buf_img, "PNG")
        image_bytes = buf_img.getvalue()
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question, "images": [image_bytes]}
        ]
        result = self.client.chat(
            model=config.VISION_MODEL,
            messages=conversation,
            stream=False,
            think=False,
            options={"temperature": 0.3},
            keep_alive=0
        )
        return result["message"]["content"]

    def image_camera(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("[CAM] Cannot red camera.")
        cap.release()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def image_monitor(self):
        return ImageGrab.grab(all_screens=True)









