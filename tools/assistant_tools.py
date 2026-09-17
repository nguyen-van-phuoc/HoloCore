import gc
import logging
import comtypes
import subprocess
import webbrowser
from yt_dlp import YoutubeDL
from datetime import datetime
from comtypes import CLSCTX_ALL
from ctypes import POINTER, cast
from pycaw.utils import AudioUtilities
from core.app_context import AppContext
from  tools.search_tool import SearchTool
from vision.vision_analyzer import ImageAnalyzer
from pycaw.api.endpointvolume import IAudioEndpointVolume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
loger = logging.getLogger(__name__)

class AssistantTools:

    def __init__(self):
        self.voice = AppContext.voice
        self.vision = ImageAnalyzer()
        self.search_tool = SearchTool()
        self._init_com()

    def _init_com(self):
        try:
            comtypes.CoInitialize()
        except Exception as e:
            print(f"[COM] {e}")

    # ------------------------------------------------------------------
    # Tools Public Methods (To be registered with MCP)
    # ------------------------------------------------------------------

    def date_time(self) -> str:
        """Returns the current system date and time."""
        now = str(datetime.now())
        print(f"[TIME] {now}")
        return now

    def set_voice_character(self, name: str) -> str:
        """
        Change the current voice for the TTS system.
        Args:
            name: The name of the voice to switch to. Supported voices: holo, mahiru, and fuwa.
        """
        if name.lower() == "holo":
            return "COMMAND:HOLO_VOICE"
        elif name.lower() == "mahiru":
            return "COMMAND:MAHIRU_VOICE"
        elif name.lower() == "fuwa":
            return "COMMAND:FUWA_VOICE"
        else:
            return "COMMAND:UNSUPPORTED_GAME"

    def play_music(self, artist: str = None, song: str = None, mood: str = None) -> str:
        """
        Find and play a song on YouTube based on the artist, song title,
        and/or desired mood.

        Args:
            artist: Artist name (optional).
            song: Song title (optional).
            mood: Desired mood or music genre (optional).
        """
        if self._is_music_playing():
            self.stop_music()

        query = " ".join(filter(None, [song, artist, mood]))
        if not query:
            return "Not enough information (artist/song/mood) to find the song."

        yt_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "extract_flat": False,
            "default_search": "ytsearch",
            "js_runtimes": {"node": {"path": r"C:\Program Files\nodejs\node.exe"}},
            "remote_components": ["ejs:github"],
        }
        try:
            with YoutubeDL(yt_opts) as ydl:
                info = ydl.extract_info(f"ytsearch:{query}", download=False)
                entries = info.get("entries", [])
                video_url = next(
                    (e.get("webpage_url") for e in entries if e and e.get("webpage_url")),
                    None,
                )
            if not video_url:
                return f"No suitable videos found for '{query}'."
            webbrowser.open(video_url)
            return f"Now playing: {video_url}"
        except Exception as e:
            return f"Error playing music:: {e}"

    def set_volume(self, volume: int) -> str:
        """
        Adjust the system volume.
        Args:
            volume: Volume value from 0 to 100.
        """
        devices = interface = endpoint = None
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            endpoint = cast(interface, POINTER(IAudioEndpointVolume))
            scalar = max(0.0, min(1.0, volume / 100))
            endpoint.SetMasterVolumeLevelScalar(scalar, None)
            return f"Volume set: {volume}%"
        except Exception as e:
            return f"Error setting volume: {e}"
        finally:
            del endpoint, interface, devices

    def stop_music(self) -> str:
        """Stop playback (close the Edge browser that is playing the video)."""
        try:
            gc.collect()
            subprocess.run(
                ["taskkill", "/f", "/im", "msedge.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "Music stopped."
        except Exception as e:
            return f"Error stopping music: {e}"

    def search_web(self, query: str) -> str:
        """Parameter (args): query (Concise, clear search keyword in English).
           Trigger condition: When the user asks about general knowledge, current events, weather, prices, or information outside the existing database.
        """
        return self.search_tool.search(query)

    def analysis_camera(self, device: str, question: str) -> str:
        """Parameters (args):
                device (Accepts one of two string values: "camera" or "monitor").
                question (string, optional): The content of the user's question or request to be sent to the vision model. Leave as an empty string "" if the user's request is general.
            Trigger conditions:
                Use "camera": When the user asks to look directly at them, or to evaluate their complexion, facial expressions, attire, or a physical object placed in front of the actual camera lens (e.g., "How do I look?", "Do I look good in this outfit today?", "Look at what I'm holding").
                Use "monitor": When the user asks to view, read, analyze, or explain content, text, or images currently displayed on the device's screen (e.g., "What is on the screen?", "Read the error message on the screen for me", "Explain the photo currently open on the screen").
        """
        loger.info(F"[QUE] {question}")
        return self.vision.analysis_camera(device, question)

    def game_play(self, name_game: str) -> str:
        """Parameters (args):
                name_game (string): The name of the game the user wants to play or open (e.g., "chess" or "cờ vua").
                Supported games: Chess, Caro (Tic-Tac-Toe).
                    If the user wants to play chess: name_game = chess
                    If the user wants to play Caro: name_game = caro

                Activation conditions:
                    When the user requests to play, start, open, or resume a game with Holo.
                    If the game already exists, simply display the current match instead of creating a new one.
                    Do not call this tool when the user is merely asking questions, chatting, or requesting an analysis of the game.
        """
        if name_game.lower() == "chess":
            return "COMMAND:OPEN_CHESS"
        elif name_game.lower() == "caro":
            return "COMMAND:OPEN_CARO"
        else:
            return "COMMAND:UNSUPPORTED_GAME"

    # ------------------------------------------------------------------
    # Internal function (prefixed with "_" -> not registered as a tool)
    # ------------------------------------------------------------------
    def _is_music_playing(self) -> bool:
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            process = session.Process
            if process and process.name().lower() == "msedge.exe":
                return True
        return False

