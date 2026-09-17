import sys
import time
import asyncio
import textwrap
import tiktoken
import traceback
import threading
from core import config
import live2d.v3 as live2d
from tts.gpt_sovits import GPTSoVits
from PyQt6.QtGui import QSurfaceFormat
from core.app_context import AppContext
from PyQt6.QtWidgets import QApplication
from live.live2d.render import Live2DWidget
from games.manager_game import GameManager
from llm.stream_processor import StreamProcessor
from PyQt6.QtCore import QTimer, QObject, pyqtSignal
from live.live2d.input_widget import TrueLiquidWidget

class GUISignals(QObject):
    open_chess_signal = pyqtSignal(str)

class PipeLine:
    def __init__(self):

        self.voice = AppContext.voice ###
        self.lip_sync_state = AppContext.lip_sync_state
        self.tts = None
        self.stt = AppContext.stt
        self.memory_long = AppContext.memory_long
        self.memory_short = AppContext.memory_short
        self.memory_rag = AppContext.memory_rag
        self.prompt_manager = AppContext.prompt_manager
        self.system_promt = self.prompt_manager.get_system_prompt()
        self.tools_prompt = self.prompt_manager.get_tools_prompt()
        self.auto_response = AppContext.auto_response
        self.client = AppContext.client
        self.last_interaction_time = time.time()
        self.params = AppContext.params
        self.mcp = AppContext.mcp
        self.idle_thread = None
        self.loop = None

        self.game_manager = GameManager()

        self.chess_window = None
        self.gui_signals = None

        self.enc = tiktoken.get_encoding("cl100k_base")

        self.tool_text = None

        self.render_ui()


    def render_ui(self):
        fmt = QSurfaceFormat()
        fmt.setAlphaBufferSize(8)
        QSurfaceFormat.setDefaultFormat(fmt)

        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        self.gui_signals = GUISignals()
        self.gui_signals.open_chess_signal.connect(self.open_game_play)

        live2d_window = Live2DWidget(self.lip_sync_state)
        input_window = TrueLiquidWidget(pipeline=self)

        self.tts = GPTSoVits(self.voice, self.lip_sync_state, live2d_window, input_window)
        AppContext.tts = self.tts

        self.tts.signals.play_motion.connect(live2d_window.play_animation)
        self.start()

        live2d_window.show()
        input_window.show()
        app.aboutToQuit.connect(self._on_quit)

        exit_code = app.exec()
        live2d.dispose()
        sys.exit(exit_code)


    def build_messages(self, user_input):
        message = []
        memories = self.memory_long.search(user_input)
        if memories:
            memory_lines = "\n".join(f"- {m}" for m in memories)
            memory_instruction = textwrap.dedent(f"""
               ### KNOWN INFORMATION ABOUT THE USER
                {memory_lines}
                - Always respond in Vietnamese.
                - Only use long-term information when useful.
                - Do not mention memory or the process of inference.
                - Prioritize the current context and avoid repetition.
                """).strip()
        else:
            memory_instruction = None
        memory_rag = self.memory_rag.search(user_input)
        if memory_rag:
            rag_instruction = textwrap.dedent(f"""
               ### RETRIEVED CONTEXT
                {memory_rag}
                - Only use the retrieved context when relevant.
                - Prioritize the current conversation if there is a conflict.
                - Do not mention RAG, memory, or the retrieval process.
                - Do not make inferences beyond the provided data and always integrate the context naturally.
                """).strip()
        else:
            rag_instruction = None
        message.append({"role": "system", "content": self.system_promt})
        if memory_instruction:
            message.append({"role": "system", "content": memory_instruction})
        if rag_instruction:
            message.append({"role": "system", "content": rag_instruction})
        message.extend(self.memory_short.get_memory_short())
        message.append({"role": "user", "content": user_input})

        text_for_tokenizer = "\n".join(m["content"] for m in message if isinstance(m, dict) and isinstance(m.get("content"), str))
        token_count = len(self.enc.encode(text_for_tokenizer))

        print(f"[Token] Number of input tokens: {token_count}")
        return message

    async def run_pipeline(self, message: list[dict]) -> str:
        processor = StreamProcessor(self.tts)
        message_mcp = [{"role": "system", "content": self.tools_prompt}, message[-1]]
        chat_options = {
            "temperature": config.TEMPERATURE,
            "repeat_penalty": config.REPEAT_PENALTY,
            "repeat_last_n": config.REPEAT_LAST_N,
            "top_k": config.TOP_K,
            "top_p": config.TOP_P,
            "num_ctx": config.NUM_CTX,
        }
        try:
            # 1) ROUTER — Retrieve the schema from the cache; do not access the MCP if it is already cached.
            ollama_tools = await self.mcp.get_tools()
            response = await self.client.chat(
                model=config.OLLAMA_MODEL,
                messages=message_mcp,
                tools=ollama_tools,
                think=False,
                options=chat_options,
            )

            tool = response["message"]["content"]
            print(f"[Tool] {tool}", end="", flush=True)

            # 2) MCP — only accessed when the router actually requests the tool
            if response.message.tool_calls:
                message.append(response.message)
                self.memory_short.add_momory_short(response.message)
                for tool_call in response.message.tool_calls:
                    try:

                        result = await self.mcp.call_tool(tool_call.function.name, tool_call.function.arguments)
                        self.tool_text = result.content[0].text if result.content else ""

                        if tool_call.function.name in ("game_play", "set_voice_character"):
                            self.command_handler(self.tool_text)
                        print(self.tool_text)

                    except Exception as e:
                        traceback.print_exc()
                        self.tool_text = f"Error executing the tool {tool_call.function.name}: {e}"
                    tool_message = {"role": "tool", "name": tool_call.function.name, "content": self.tool_text}
                    message.append(tool_message)
                    self.memory_short.add_momory_short(tool_message)
            else:
                print()

            # 3) FINAL — completely separated from the MCP lifecycle
            stream = await self.client.chat(
                model=config.OLLAMA_MODEL,
                messages=message,
                tools=None,
                think=False,
                stream=True,
                options=chat_options,
            )
            print("[Holo] ", end="", flush=True)
            async for token in stream:
                chunk = token["message"]["content"]
                if chunk:
                    processor.process_token(chunk)
            processor.flush()
            self.memory_short.add_momory_short({"role": "assistant", "content": processor.full_response})
        except Exception as e:
            traceback.print_exc()
            print(f"[LLM ERROR] {e}")

        self.tts.wait_until_done()
        return processor.full_response

    def monitor_idle_time(self):
        while True:
            time.sleep(10)
            if self.loop is None:
                # main() hasn't run yet, so there is no event loop to schedule tasks on.
                continue
            if time.time() - self.last_interaction_time > config.IDLE_THRESHOLD and self.auto_response.is_auto:
                print("\n[Holo] Actively chatting...")
                instruction = self.auto_response.get_random_instruction()
                messages = [
                    {"role": "system", "content": self.system_promt},
                    {"role": "system", "content": instruction},
                ]
                future = asyncio.run_coroutine_threadsafe(
                    self.run_pipeline(messages),
                    self.loop
                )
                response = future.result()
                self.auto_response.lever = self.auto_response.lever + 1
                self.memory_short.add_momory_short({"role": "assistant", "content": response})
                self.memory_long.add_assistant_turn(response)
                self.last_interaction_time = time.time()
                print("[User] ", end="", flush=True)

    def help(self):
        print("=" * 60)
        print(" Holo AI — Ollama + TTS + STT+ Vision")
        print("=" * 60)
        print("Nhập 'q' hoặc 'exit' hoặc 'quit' để thoát.")
        print("Các chức năng hiện có: ")
        print("0. /add path file - thêm tri thức LLM")
        print("2. /mic boolean - Sử dụng mic để chat" )
        print("3. play_music — Phát nhạc")
        print("4. set_volume — Chỉnh âm lượng")
        print("5. stop_music — Dừng nhạc")
        print("6. set_voice_character — Đổi giọng nói")
        print("7. search_web — Tra cứu Internet")
        print("8. date_time — Lấy ngày giờ hiện tại")
        print("9. analysis_camera — Nhìn/Phân tích qua camera hoặc màn hình")
        print("Lưu ý: 2-9 chỉ có thể điều hiển thông qua LLM.")

    def start(self):
        """Create exactly one background event loop that persists for the app's entire lifecycle
        and runs in a dedicated thread (since Qt occupies the main thread). All coroutines (main(),
        idle auto-response, shutdown()) must be scheduled onto this loop using run_coroutine_threadsafe—do
        NOT use asyncio.run() for individual turns, as each call creates a new loop;
        this causes the MCP session (which is intended to persist across multiple chat turns)
        to become misaligned with the task/loop, resulting in a "cancel scope in a different task" error upon shutdown."""
        self.last_interaction_time = time.time()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.idle_thread = threading.Thread(target=self.monitor_idle_time, daemon=True)
        self.idle_thread.start()
        self.tts.start()

    def _on_quit(self):
        """Connect to app.aboutToQuit: close the MCP properly before exiting.
        shutdown() must run on self.loop—the only place where the MCP connection is opened—
        before stopping the background loop."""
        try:
            future = asyncio.run_coroutine_threadsafe(self.shutdown(), self.loop)
            future.result(timeout=5)
        except Exception as e:
            print(f"[WARNING] Forced closure due to timeout or error: {e}")
            traceback.print_exc()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            print("[APP] Safely shut down!")

    async def shutdown(self):
        """Closes the MCP and other resources. It is always called via `_on_quit` on the correct `self.loop`,
        thereby avoiding "cancel-scope-in-different-task" errors during application shutdown."""
        try:
            self.tts.stop()
            self.memory_long.close()
            if hasattr(self.mcp, 'close'):
                await asyncio.wait_for(self.mcp.close(), timeout=3.0)
        except Exception as e:
            print(f"[SHUTDOWN] {e}")

    def request_shutdown(self):
        print("[APP] Sending request to close the application......")
        QTimer.singleShot(0, QApplication.instance().quit)


    def command_handler(self, command: str):
        if "OPEN" in command:
            self.gui_signals.open_chess_signal.emit(command)
            self.tool_text = f"Game opened: {command}"
        elif "VOICE" in command:
            self.voice.set_voice_character(command)
            self.tool_text = f"The character's voice has been changed to: {command}"
        else:
            self.tool_text = "[Error] Command not supported!"
            print("[COM] Command not supported!")

    def open_game_play(self, text: str):
        self.auto_response.is_auto = False
        self.game_manager.game_play(text)


    async def main(self, user_input: str):
        try:
            time0 = time.time()
            self.last_interaction_time = time.time()
            self.auto_response.lever = 1
            if not user_input.strip():
                return
            if user_input.lower() in ("q", "exit", "quit"):
                threading.Thread(target=self.request_shutdown, daemon=True).start()
                return
            if user_input.startswith("/add "):
                file_path = user_input[5:].strip()
                self.memory_rag.ingest_document(file_path)
                return
            if user_input.lower() in ("h", "help"):
                self.help()
                return

            messages = self.build_messages(user_input)
            time1 = time.time()
            print(f"[Time] {time1 - time0}")
            response = await self.run_pipeline(messages)
            if response:
                self.memory_short.add_momory_short({"role": "user", "content": user_input})
                self.memory_short.add_momory_short({"role": "assistant", "content": response})
                self.memory_long.add_user_turn(user_input)
                self.memory_long.add_assistant_turn(response)
                print()
        finally:
            print("Done!\n")