from core.claude import Claude
from mcp_client import MCPClient
from core.tools import ToolManager
from anthropic.types import MessageParam


class Chat:
    MAX_TOOL_ROUNDS = 20

    def __init__(self, claude_service: Claude, clients: dict[str, MCPClient]):
        self.claude_service: Claude = claude_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[MessageParam] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        final_text_response = ""
        history_len = len(self.messages)

        try:
            await self._process_query(query)

            for _ in range(self.MAX_TOOL_ROUNDS):
                response = self.claude_service.chat(
                    messages=self.messages,
                    tools=await ToolManager.get_all_tools(self.clients),
                )

                self.claude_service.add_assistant_message(
                    self.messages, response
                )

                if response.stop_reason == "tool_use":
                    print(self.claude_service.text_from_message(response))
                    tool_result_parts = (
                        await ToolManager.execute_tool_requests(
                            self.clients, response
                        )
                    )

                    self.claude_service.add_user_message(
                        self.messages, tool_result_parts
                    )
                else:
                    final_text_response = (
                        self.claude_service.text_from_message(response)
                    )
                    break
            else:
                raise RuntimeError(
                    f"Stopped after {self.MAX_TOOL_ROUNDS} tool rounds"
                )
        except BaseException:
            # Drop the partial turn so history stays valid for the next query.
            del self.messages[history_len:]
            raise

        return final_text_response
