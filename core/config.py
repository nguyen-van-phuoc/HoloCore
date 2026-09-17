# Ollama

# Main chat / tool decision model
OLLAMA_MODEL = "qwen3:4b-instruct"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
TOOLS = None

# Vision model
VISION_MODEL = "qwen3-vl:2b-instruct"

# RAG embedding model
EMBED_MODEL_RAG = "nomic-embed-text"
PATH_RAG_DB = "memory/rag"

# Memory
MEMORY_INDEX = "memory/long/memory.index"
MEMORY_JSON = "memory/long/memory.json"
EMBED_MODEL_MEMORY = "paraphrase-multilingual-MiniLM-L12-v2"

# AI
NUM_CTX =  4096
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.9
REPEAT_PENALTY = 1.2
REPEAT_LAST_N = 128

# STT
STT_LANGUAGE = "vi"

# Idle
IDLE_THRESHOLD = 300

# Audio
AUDIO_TAIL_PADDING_MS = 150