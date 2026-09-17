import gc
import sys
import torch
import core.config as config
from ollama import AsyncClient
from mcp import StdioServerParameters
from memory.memory_long import Memory
from llm.auto_response import AutoResponse
from tts.manager_voice import VoiceManager
from memory.momory_short import MemoryShort
from llm.prompt_manager import PromptManager
from memory.memory_rag import MemoryRAGSystem
from tools.mcp_manager import MCPManager
from live.live2d.lip_sync_state import LipSyncState

class AppContext:
    voice = None
    lip_sync_state = None
    tts = None
    stt = None
    memory_long = None
    memory_short = None
    memory_rag = None
    prompt_manager = None
    auto_response = None
    client = None
    params = None
    mcp = None

    @classmethod
    def init(cls):
        cls.collect()
        cls.voice = VoiceManager()
        cls.lip_sync_state = LipSyncState()
        cls.stt = None
        cls.memory_long = Memory(index_file=config.MEMORY_INDEX, data_file=config.MEMORY_JSON, model_name=config.EMBED_MODEL_MEMORY)
        cls.memory_short = MemoryShort()
        cls.memory_rag = MemoryRAGSystem(embed_model=config.EMBED_MODEL_RAG, path=config.PATH_RAG_DB)
        cls.prompt_manager = PromptManager()
        cls.auto_response = AutoResponse()
        cls.client = AsyncClient(host=config.OLLAMA_BASE_URL)
        cls.params = StdioServerParameters(command=sys.executable, args=["-m", "tools.server"])
        cls.mcp = MCPManager(cls.params)

    @classmethod
    def collect(cls) -> None:
        gc.collect()
        print("[RAM] Cleared")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[VRA] Cleared")



