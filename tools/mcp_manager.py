import asyncio
import traceback
from typing import Any
from mcp import ClientSession, stdio_client


class MCPManager:
    """Manages an Model Context Protocol (MCP) server lifecycle over stdio connection,

    handling background task runner, tool discovery/caching, and request queuing.
    """

    def __init__(self, params: Any):
        """Initialize the MCP Manager with connection parameters."""
        self.params = params
        self._requests: asyncio.Queue = asyncio.Queue()
        self._tools_cache: list[dict] | None = None
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _ensure_started(self) -> None:
        """Ensure the background worker task is active and running."""
        if self._task is None or self._task.done():
            self._ready = asyncio.Event()
            self._tools_cache = None
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        """Main background loop handling stdio communication, initialization,

        tool caching, and tool execution requests.
        """
        try:
            async with stdio_client(self.params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()

                    # Fetch and format tools into standard OpenAI function specification
                    mcp_tools = (await session.list_tools()).tools
                    self._tools_cache = [
                        {
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.inputSchema,
                            },
                        }
                        for t in mcp_tools
                    ]
                    self._ready.set()

                    # Process incoming request queue
                    while True:
                        name, arguments, future = await self._requests.get()

                        # Stop signal received (see close())
                        if name is None:
                            future.set_result(None)
                            return

                        try:
                            result = await session.call_tool(name, arguments)
                            future.set_result(result)
                        except Exception as e:
                            # Session might be terminated (subprocess crash, closed pipe, etc.).
                            # Avoid auto-retrying here to prevent executing tools with side-effects
                            # twice (e.g., play_music, set_volume). Exit worker loop to cleanup
                            # connection; subsequent call_tool() calls will reconnect automatically.
                            traceback.print_exc()
                            future.set_exception(e)
                            return
        except Exception:
            traceback.print_exc()
        finally:
            self._ready.set()

    async def get_tools(self) -> list[dict]:
        """Retrieve the list of cached MCP tools (in OpenAI function calling format)."""
        await self._ensure_started()
        await self._ready.wait()
        return self._tools_cache or []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Execute a specific tool on the MCP server.

        :param name: Name of the tool to execute.
        :param arguments: Arguments dictionary passed to the tool.
        :return: Execution result from the MCP tool.
        """
        await self._ensure_started()
        await self._ready.wait()

        if self._task.done():
            raise RuntimeError(f"Failed to connect to MCP server to call tool '{name}'")

        future = asyncio.get_running_loop().create_future()
        await self._requests.put((name, arguments, future))
        return await future

    async def close(self) -> None:
        """Gracefully close the MCP session and terminate the background worker task."""
        if self._task is None or self._task.done():
            return

        future = asyncio.get_running_loop().create_future()
        await self._requests.put((None, None, future))
        await self._task