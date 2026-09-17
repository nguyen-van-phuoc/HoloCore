import time
import requests
import threading

class Connection:
    def __init__(self):
        self.running = False
        self.online = False
        self.check_connection()
        self.text = ""
        self.start()

    def check_connection(self):
        try:
            requests.get("https://www.google.com", timeout=2)
            self.online = True
        except requests.RequestException:
            self.online = False

    def loop_connection(self):
        while self.running:
            try:
                requests.get("https://www.google.com", timeout=2)
                self.online = True
            except requests.RequestException:
                self.online = False
            except Exception as e:
                print(f"[CON] {e}")
                self.stop()
            time.sleep(3)

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.loop_connection, daemon=True).start()

    def stop(self):
        self.running = False

    def is_online(self):
        return self.online