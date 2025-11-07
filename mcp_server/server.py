#!/usr/bin/env python3
"""
MCP Server for CLI-based LLM context management.
Supports project-based conversation storage with session tracking.

Usage:
    python server.py

Configuration in Claude Desktop:
    {
        "mcpServers": {
            "context-manager": {
                "command": "python",
                "args": ["/path/to/mcp_server/server.py"],
                "env": {
                    "CONTEXT_STORAGE_DIR": "~/.mcp_contexts"
                }
            }
        }
    }
"""
import os
import sys
import asyncio
import logging
from typing import Any, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from context_manager import ContextManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser('~/.mcp_contexts/server.log')),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-context-server")


class ContextMCPServer:
    """MCP Server for context management."""

    def __init__(self):
        self.app = Server("context-manager")
        self.context_manager = ContextManager(
            storage_dir=os.getenv("CONTEXT_STORAGE_DIR", "~/.mcp_contexts")
        )

        # Register handlers
        self.app.list_tools()(self.list_tools)
        self.app.call_tool()(self.call_tool)

        logger.info("Context MCP Server initialized")

    async def list_tools(self) -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="store_conversation",
                description="Store a conversation message in the current session. "
                           "Automatically tracks project and session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant", "system"],
                            "description": "Message role"
                        },
                        "content": {
                            "type": "string",
                            "description": "Message content"
                        },
                        "token_count": {
                            "type": "integer",
                            "description": "Token count for this message",
                            "default": 0
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata (model, temperature, etc.)",
                            "default": {}
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (auto-detected from cwd if not provided)"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Current working directory for project detection"
                        }
                    },
                    "required": ["role", "content"]
                }
            ),
            Tool(
                name="get_context",
                description="Get conversation history for the current session. "
                           "Returns messages in chronological order.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of messages to return",
                            "default": None
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (auto-detected if not provided)"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Specific session ID (current session if not provided)"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Current working directory"
                        }
                    }
                }
            ),
            Tool(
                name="search_context",
                description="Search conversation history by content. "
                           "Searches within current project.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Limit search to specific session (all sessions if not provided)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Current working directory"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="clear_session",
                description="End current session (e.g., when token limit reached). "
                           "Marks session as inactive. Use when starting a new conversation.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session to end (current session if not provided)"
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Current working directory"
                        }
                    }
                }
            ),
            Tool(
                name="list_projects",
                description="List all projects with stored conversations.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_stats",
                description="Get statistics for current session or project.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["session", "project"],
                            "description": "Type of stats to retrieve",
                            "default": "session"
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (for session stats)"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Current working directory"
                        }
                    }
                }
            ),
            Tool(
                name="start_session",
                description="Explicitly start a new session for the current project. "
                           "Usually called automatically, but can be used to force a new session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (auto-detected if not provided)"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Custom session ID (auto-generated if not provided)"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Current working directory"
                        }
                    }
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Any) -> list[TextContent]:
        """Handle tool calls."""
        try:
            logger.info(f"Tool called: {name} with args: {arguments}")

            # Handle cwd if provided
            if "cwd" in arguments:
                original_cwd = os.getcwd()
                os.chdir(arguments["cwd"])
            else:
                original_cwd = None

            result = None

            if name == "store_conversation":
                result = await self._store_conversation(arguments)

            elif name == "get_context":
                result = await self._get_context(arguments)

            elif name == "search_context":
                result = await self._search_context(arguments)

            elif name == "clear_session":
                result = await self._clear_session(arguments)

            elif name == "list_projects":
                result = await self._list_projects(arguments)

            elif name == "get_stats":
                result = await self._get_stats(arguments)

            elif name == "start_session":
                result = await self._start_session(arguments)

            else:
                result = {"error": f"Unknown tool: {name}"}

            # Restore cwd
            if original_cwd:
                os.chdir(original_cwd)

            logger.info(f"Tool {name} completed successfully")

            return [TextContent(
                type="text",
                text=str(result)
            )]

        except Exception as e:
            logger.error(f"Error in tool {name}: {str(e)}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]

    async def _ensure_session(
        self,
        project_id: Optional[str] = None
    ) -> str:
        """Ensure a session is active."""
        if not self.context_manager.current_session_id:
            return self.context_manager.start_session(project_id=project_id)
        return self.context_manager.current_session_id

    async def _store_conversation(self, args: dict) -> dict:
        """Store a conversation message."""
        await self._ensure_session(args.get("project_id"))

        self.context_manager.store_message(
            role=args["role"],
            content=args["content"],
            metadata=args.get("metadata"),
            token_count=args.get("token_count", 0)
        )

        return {
            "status": "success",
            "project_id": self.context_manager.current_project_id,
            "session_id": self.context_manager.current_session_id
        }

    async def _get_context(self, args: dict) -> dict:
        """Get conversation context."""
        await self._ensure_session(args.get("project_id"))

        messages = self.context_manager.get_context(
            limit=args.get("limit"),
            session_id=args.get("session_id")
        )

        return {
            "messages": messages,
            "count": len(messages),
            "project_id": self.context_manager.current_project_id,
            "session_id": args.get("session_id") or self.context_manager.current_session_id
        }

    async def _search_context(self, args: dict) -> dict:
        """Search conversation history."""
        await self._ensure_session(args.get("project_id"))

        results = self.context_manager.search_context(
            query=args["query"],
            session_id=args.get("session_id"),
            limit=args.get("limit", 10)
        )

        return {
            "results": results,
            "count": len(results),
            "query": args["query"]
        }

    async def _clear_session(self, args: dict) -> dict:
        """Clear/end current session."""
        if args.get("project_id"):
            await self._ensure_session(args["project_id"])

        session_id = args.get("session_id") or self.context_manager.current_session_id

        self.context_manager.end_session(session_id)

        return {
            "status": "success",
            "message": f"Session {session_id} ended",
            "session_id": session_id
        }

    async def _list_projects(self, args: dict) -> dict:
        """List all projects."""
        projects = self.context_manager.list_projects()

        return {
            "projects": projects,
            "count": len(projects)
        }

    async def _get_stats(self, args: dict) -> dict:
        """Get statistics."""
        stats_type = args.get("type", "session")

        await self._ensure_session(args.get("project_id"))

        if stats_type == "session":
            stats = self.context_manager.get_session_stats(
                session_id=args.get("session_id")
            )
        else:  # project
            stats = self.context_manager.get_project_stats(
                project_id=args.get("project_id")
            )

        return stats or {"error": "No stats available"}

    async def _start_session(self, args: dict) -> dict:
        """Start a new session."""
        session_id = self.context_manager.start_session(
            project_id=args.get("project_id"),
            session_id=args.get("session_id")
        )

        return {
            "status": "success",
            "session_id": session_id,
            "project_id": self.context_manager.current_project_id
        }

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting MCP server with STDIO transport")

        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream,
                write_stream,
                self.app.create_initialization_options()
            )


def main():
    """Main entry point."""
    server = ContextMCPServer()

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
