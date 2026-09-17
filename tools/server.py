import sys
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from tools.assistant_tools import AssistantTools
logging.getLogger("tools").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class AssistantMCPServer:
    def __init__(self, name: str = "voice-assistant"):
        self.tools = AssistantTools()
        self.mcp = FastMCP(name)
        self._register_tools()

    def _register_tools(self):
        self.mcp.add_tool(self.tools.search_web)
        self.mcp.add_tool(self.tools.play_music)
        self.mcp.add_tool(self.tools.set_volume)
        self.mcp.add_tool(self.tools.stop_music)
        self.mcp.add_tool(self.tools.set_voice_character)
        self.mcp.add_tool(self.tools.date_time)
        self.mcp.add_tool(self.tools.analysis_camera)
        self.mcp.add_tool(self.tools.game_play)

    def run(self, transport: str = "stdio"):
        self.mcp.run(transport=transport)

if __name__ == "__main__":
    server = AssistantMCPServer()
    server.run(transport="stdio")