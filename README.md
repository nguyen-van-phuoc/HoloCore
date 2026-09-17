# HoloCore

**HoloCore** is a modular, local-first AI voice assistant designed to run primarily on the user's own machine.

It combines a local LLM, offline speech recognition, voice synthesis, Live2D, memory, RAG, vision, MCP tools, games, and system controls into a single extensible pipeline.

The project is designed around a simple principle:

> **Keep the AI local, modular, and under the user's control.**

---

## 🎬 Demo

[▶️ HoloCore Demo](https://drive.google.com/file/d/1gwGqRelhS2m_1T9f0ekDii0LuJ82nP9J/view?usp=drive_link)

---

## ✨ Features

* 🧠 **Local LLM** powered by Ollama
* 🎙️ **Offline STT** for speech recognition
* 🔊 **Local TTS** with GPT-SoVITS
* 🎭 **Live2D avatar** with motion control and lip sync
* 🛠️ **MCP tool system** for modular tool execution
* 💾 **Short-term and long-term memory**
* 📚 **Local RAG** for document knowledge
* 👁️ **Vision / camera analysis**
* ♟️ **Chess and Gomoku (Caro)**
* 🎵 **Music and system controls**
* 🌐 **Optional web search**
* ⚡ **Streaming LLM responses**
* 🔌 **Modular architecture**

---

## 🏗️ Architecture

HoloCore uses a pipeline architecture that separates context preparation, tool execution, response generation, and voice output.

```text
                          HoloCore
                             │
                             ▼
                            STT
                             │
                             │ User Text
                             ▼
                      build_messages()
                             │
               ┌─────────────┼─────────────┐
               │             │             │
               ▼             ▼             ▼
         Long-Term       Local RAG    Short-Term
          Memory                       Memory
               │             │             │
               └─────────────┼─────────────┘
                             │
                             ▼
                      System / Prompt
                             │
                             ▼
                    Complete Messages
                             │
                             ▼
                       run_pipeline()
                             │
                             ▼
                    Ollama - Tool Decision
                             │
                  ┌──────────┴───────────┐
                  │                      │
               No Tool               Tool Call
                  │                      │
                  │                      ▼
                  │                     MCP
                  │                      │
                  │                      ▼
                  │                 Tool Result
                  │                      │
                  └──────────┬───────────┘
                             ▼
                  Ollama - Main Response
                             │
                             │ Streaming Text
                             ▼
                            TTS
                             │
                             │ Audio
                             ▼
                     Live2D / Lip Sync
```

The LLM is used in two stages:

1. **Tool decision** - determines whether an MCP tool is required.
2. **Main response** - generates the final response after tool execution.

This keeps tool execution separate from response generation and makes the pipeline easier to extend.

---

## 🧩 Core Components

| Component             | Responsibility                                      |
| --------------------- | --------------------------------------------------- |
| **STT**               | Converts microphone input into text                 |
| **build_messages()**  | Builds the complete LLM context                     |
| **Long-Term Memory**  | Stores persistent information                       |
| **Short-Term Memory** | Maintains recent conversation context               |
| **RAG**               | Retrieves relevant information from local documents |
| **Prompt System**     | Defines assistant behavior and instructions         |
| **Ollama**            | Runs local LLM, vision and embedding models         |
| **MCP**               | Provides modular tool execution                     |
| **TTS**               | Converts generated text into speech                 |
| **Live2D**            | Displays avatar, motion and lip synchronization     |
| **Vision**            | Handles image and camera analysis                   |
| **Games**             | Provides chess and Gomoku functionality             |

---

## 📁 Project Structure

```text
HoloCore/
│
├── core/
│   └── config.py              # Global configuration
│
├── llm/
│   └── ...                     # LLM Processing
│
├── stt/
│   └── ...                     # Speech recognition
│
├── tts/
│   └── ...                     # Text-to-speech
│
├── memory/
│   └── ...                     # Memory and RAG systems
│
├── tools/
│   └── ...                     # Tool-related components
│
├── vision/
│   └── ...                     # Vision / camera analysis
│
├── games/
│   └── ...                     # Chess / Gomoku
│
├── live/
│   └── ...                     # Live2D integration
│
├── audio/
│   └── ...                     # Audio resources
│
├── images/
│   └── ...                     # Image resources
│
├── pipeline.py                 # Main AI pipeline
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration
└── README.md
```

---

# 🚀 Installation

This section contains the complete setup required to run HoloCore.

## ⚙️ Requirements

### Operating System

* Windows

### Software

* Python **3.10+**
* Ollama
* FFmpeg
* GPT-SoVITS
* Stockfish
* NVIDIA GPU recommended

### AI Components

HoloCore uses separate local models for:

* Chat / tool reasoning
* Vision
* RAG embeddings
* Speech recognition
* Voice synthesis

---

## 1. Clone the Repository

```bash
git clone https://github.com/nguyen-van-phuoc/HoloCore.git
cd HoloCore
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

HoloCore uses Ollama for local model inference.

Download and install: [Ollama](https://ollama.com/download)

Verify the installation:

```powershell
ollama --version
```

---

## 5. Download Ollama Models

Download the models required by HoloCore:

```powershell
ollama pull qwen3:4b-instruct
ollama pull qwen3-vl:2b-instruct
ollama pull nomic-embed-text
```

Check installed models:

```powershell
ollama list
```

---

## 6. Install FFmpeg

HoloCore requires **FFmpeg** for audio processing.

Download FFmpeg: [FFmpeg](https://ffmpeg.org/download.html)

Extract FFmpeg and add its `bin` directory to the system **PATH**.

Verify:

```powershell
ffmpeg -version
```

---

## 7. Install GPT-SoVITS

HoloCore uses **GPT-SoVITS** as its local TTS backend.

Official project: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

Download the required pretrained models: [GPT-SoVITS pretrained models](https://huggingface.co/lj1995/GPT-SoVITS)

Place the pretrained components according to the GPT-SoVITS version being used.

Example:

```text
GPT-SoVITS/
└── GPT_SoVITS/
    └── pretrained_models/
        ├── chinese-hubert-base
        ├── chinese-roberta-wwm-ext-large
        └── gsv-v2final-prettrained
```

> GPT-SoVITS versions use different model files. Always use the pretrained models compatible with the version installed in your environment.

---

## 8. Install Stockfish

HoloCore uses **Stockfish** as the local chess engine.

Download Stockfish: [Stockfish](https://stockfishchess.org/download/)

Extract the archive and place the executable in:

```text
games/
└── chess/
    └── stockfish/
        └── <executable>
```

Example:

```text
games/chess/stockfish/stockfish-windows-x86-64-avx2.exe
```

---

### 9. NVIDIA CUDA — Optional

If you are using an **NVIDIA GPU**, install the CUDA-enabled PyTorch packages for GPU acceleration.

First, verify that your NVIDIA driver is working:

```powershell
nvidia-smi
```

Then install the CUDA-enabled PyTorch build:

Example:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA support:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

If CUDA is available, the output should show:

```text
CUDA: True
GPU: NVIDIA ...
```

> If you do not have an NVIDIA GPU, skip this step and install the normal dependencies instead.

> The CUDA version used by PyTorch must match a supported PyTorch CUDA build. Check the official PyTorch installation page if you need a different CUDA version.

---

## 🔧 Configuration

Most global model and runtime configuration is centralized in:

```text
core/config.py
```

Example:

```python
# Main chat / tool decision model
OLLAMA_MODEL = "qwen3:4b-instruct"

# Vision model
VISION_MODEL = "qwen3-vl:2b-instruct"

# RAG embedding model
EMBED_MODEL_RAG = "nomic-embed-text"
```

The model names must exactly match the models installed in Ollama.

You can replace these models with other compatible Ollama models without changing the overall architecture.

The architecture intentionally avoids hard-coding model choices throughout the project.

---

## ▶️ Running HoloCore

After completing the installation and configuration:

```powershell
python main.py
```

The application starts the main HoloCore runtime and initializes the required subsystems.

---

# 🧠 Memory & Context System

HoloCore uses three independent context sources:

* **Long-Term Memory** - persistent user information across sessions.
* **Short-Term Memory** - recent conversation context.
* **Local RAG** - relevant knowledge retrieved from local documents.

Long-term memory and RAG are independent systems with different purposes.

## 📨 Message Building

`build_messages()` retrieves and combines the required context with the user input and system prompt before passing it to the main pipeline.

```text
User Input
    │
    ▼
build_messages()
    │
    ├── Long-Term Memory
    ├── Short-Term Memory
    ├── Local RAG
    └── System Prompt
    │
    ▼
messages
    │
    ▼
run_pipeline(messages)
```

This separates **context retrieval** from **LLM execution and tool handling**.

---

# 🛠️ MCP Tool System

HoloCore uses **MCP (Model Context Protocol)** to provide tools to the local LLM.

The MCP layer allows tools to be added without tightly coupling their implementation to the main pipeline.

Available tools include:

| Tool            | Purpose                    |
| --------------- | -------------------------- |
| **Web Search**  | Search the internet        |
| **Music**       | Play and control music     |
| **Volume**      | Control system volume      |
| **Voice**       | Change voice configuration |
| **Camera**      | Analyze camera input       |
| **Date & Time** | Retrieve date and time     |
| **Chess**       | Play chess                 |
| **Gomoku**      | Play Gomoku / Caro         |

## MCP Pipeline

```text
messages
   │
   ▼
Ollama
   │
   ├──────── No Tool ────────┐
   │                         │
   │ Tool Call               │
   ▼                         │
  MCP                        │
   │                         │
   ▼                         │
Tool Result                  │
   │                         │
   └─────────────┬───────────┘
                 ▼
         Ollama Main Response
```

The LLM does not directly execute tools.

Tool calls are handled through the MCP layer, which executes the requested tool and returns the result to the LLM for the final response.

---

# 👁️ Vision

HoloCore supports local vision models through Ollama.

The vision model is configured independently from the main chat model:

```python
VISION_MODEL = "qwen3-vl:2b-instruct"
```

Vision functionality can be used for:

* Image analysis
* Camera analysis
* Visual tool interactions

The vision model does not need to be the same model used for normal conversation.

---

# 🎙️ Speech Recognition

HoloCore supports offline speech recognition.

The STT subsystem is separated from the main LLM pipeline:

```text
Microphone
    │
    ▼
STT
    │
    ▼
Text
    │
    ▼
build_messages()
```

Configure the STT language in `core/config.py`:

```python
STT_LANGUAGE = "vi"
```

Set `STT_LANGUAGE` to the appropriate language code for the language being recognized.

This allows different STT engines or models to be integrated without changing the rest of the architecture.

---

# 🔊 TTS - GPT-SoVITS

HoloCore uses **GPT-SoVITS** for local voice synthesis.

GPT-SoVITS provides zero-shot / reference-based voice cloning and is integrated as the TTS backend.

The required installation and pretrained models are described in the [Installation](#-installation) section.

---

# 🎭 Live2D

HoloCore integrates a Live2D avatar for visual interaction, including motion control and audio-driven lip synchronization.

```text
LLM Response
     │
     ├── Dialogue
     └── Motion
          │
          ▼
         TTS
          │
          ▼
        Audio
          │
          ▼
       Lip Sync
          │
          ▼
        Live2D
```

Motion markers are embedded directly into streamed responses and processed by the avatar system.

---

# ♟️ Games

HoloCore provides local game functionality through MCP.

Currently supported games:

* Chess
* Gomoku / Caro

Chess uses the local **Stockfish** engine for move analysis.

Games are isolated from the main LLM pipeline and exposed through MCP tools.

The required Stockfish installation is described in the [Installation](#-installation) section.

---

# ⚡ Streaming

HoloCore streams the LLM response directly into dialogue processing and TTS.

```text
Ollama
   │
   ▼
Pipeline → Dialogue → TTS → Audio
```

This reduces perceived response latency and allows the assistant to begin speaking before the complete response has been generated.

---

# 🌐 Optional Web Search

HoloCore can optionally provide web search through an MCP tool.

Web search is not required for the core local AI pipeline.

The assistant can therefore operate primarily offline while still supporting external search when explicitly enabled.

Configure the required API key in:

```text
.env
```

Example:

```env
SERPER_API_KEY=your_api_key_here
```

---

# 🧱 Design Principles

HoloCore is built around several architectural principles.

### Local-first

The core AI functionality is designed to run locally.

### Modular

STT, TTS, LLM, memory, RAG, vision, games, and tools are separated into independent components.

### Tool isolation

External capabilities are exposed through MCP instead of being tightly coupled to the LLM implementation.

### Model independence

Chat, vision, and embedding models can be configured independently.

### Extensibility

New tools, models, memory systems, or interfaces can be added without redesigning the entire application.

### Streaming-first

The response pipeline is designed around streaming generation to reduce latency.

---

# 📌 Project Status

HoloCore is currently under active development.

The architecture and individual components may change as the project evolves.

---

# 📄 License

This project is currently under development and does not yet have a defined open-source license.
