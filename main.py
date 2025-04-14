import argparse
import json
import logging
import anyio
import os
import sys
import anthropic
import base64
import mimetypes
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from jsonschema import validate
from rich.console import Console
from rich.syntax import Syntax
from rich import print as rprint
import traceback
from pydantic import BaseModel, model_validator, TypeAdapter
from typing import Literal, Union

if os.name == "nt":
    sys.stderr = open(os.devnull, "w")

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("python_mcp_client")
console = Console()

SUPPORTED_FILE_TYPES = {
    "application/pdf": "document",
    "image/png": "image",
    "image/jpeg": "image",
    "image/webp": "image"
}


class BaseLLMConfig(BaseModel):
    provider: Literal["anthropic", "vertexAnthropic"]
    model: str

    @model_validator(mode="before")
    @classmethod
    def validate_provider_fields(cls, values):
        provider = values.get("provider")
        if provider == "vertexAnthropic":
            if not values.get("project_id") or not values.get("region"):
                raise ValueError(
                    "vertexAnthropic requires project_id and region")
        return values


class AnthropicConfig(BaseLLMConfig):
    provider: Literal["anthropic"]


class VertexAnthropicConfig(BaseLLMConfig):
    provider: Literal["vertexAnthropic"]
    project_id: str
    region: str


LLMConfig = Union[AnthropicConfig, VertexAnthropicConfig]


async def run_session(session: ClientSession, llm_config: LLMConfig, args):
    console.print(
        "\n[bold cyan]💬 Chat started! Type 'exit' or 'quit' to leave.[/bold cyan]")

    logger.info("Initializing MCP session...")
    await session.initialize()

    logger.info("Listing tools from MCP server...")
    results = await session.list_tools()

    if not results or not hasattr(results, 'tools') or not results.tools:
        console.print(
            "[bold red]❌ No tools found on the MCP server or results format unexpected. Cannot proceed.[/bold red]")
        return

    processed_tools = []
    tool_map = {}
    for tool in results.tools:
        try:
            raw_tool = tool.model_dump() if hasattr(tool, 'model_dump') else tool.dict()
            tool_name = raw_tool.get('name', 'UNKNOWN_TOOL')
            input_schema = raw_tool.pop("inputSchema", None)
            if isinstance(input_schema, dict) and input_schema:
                raw_tool["input_schema"] = input_schema
                processed_tools.append(raw_tool)
                tool_map[tool_name] = raw_tool
        except Exception as e:
            logger.warning("Skipping tool due to error: %s", e)

    if not processed_tools:
        console.print(
            "[bold red]❌ No valid tools to process. Exiting.[/bold red]")
        return

    if llm_config.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[bold red]❌ Missing ANTHROPIC_API_KEY.[/bold red]")
            return
        client = anthropic.AsyncAnthropic(api_key=api_key)

    elif llm_config.provider == "vertexAnthropic":
        from anthropic import AsyncAnthropicVertex
        client = AsyncAnthropicVertex(
            project_id=llm_config.project_id,
            region=llm_config.region,
        )

    model = llm_config.model
    conversation_history = []

    file_attachment = None
    if args.file:
        file_path = args.file
        if not os.path.isfile(file_path):
            console.print(
                f"[bold red]❌ File not found: {file_path}[/bold red]")
            return

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or mime_type not in SUPPORTED_FILE_TYPES:
            console.print(
                f"[bold red]❌ Unsupported file type: {mime_type or 'unknown'}[/bold red]\n"
                "Supported types: PDF, PNG, JPG, WEBP"
            )
            return

        with open(file_path, "rb") as f:
            file_data = f.read()

        source = {
            "type": "base64",
            "data": base64.b64encode(file_data).decode("utf-8"),
            "media_type": mime_type
        }

        file_type = SUPPORTED_FILE_TYPES[mime_type]
        file_attachment = {
            "type": file_type,
            "source": source
        }

    file_attachment_used = False

    while True:
        print("\033[1;35mYou:\033[0m ", end="", flush=True)
        user_input_raw = await anyio.to_thread.run_sync(input)
        user_query = user_input_raw.strip()

        if user_query.lower() in {"exit", "quit"}:
            console.print("\n[bold cyan]👋 Exiting chat.[/bold cyan]")
            break

        if not user_query:
            continue

        user_content = [{"type": "text", "text": user_query}]
        if not file_attachment_used and file_attachment:
            user_content.append(file_attachment)
            file_attachment_used = True

        conversation_history.append({"role": "user", "content": user_content})

        while True:
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=conversation_history,
                    tools=processed_tools
                )
            except Exception as e:
                console.print(
                    f"[bold red]❌ LLM request failed:[/bold red] {e}")
                break

            assistant_response_content_blocks = []
            tool_calls_to_make = []
            final_text_parts = []

            if not response or not response.content:
                console.print(
                    "[bold red]⚠️ Empty response from LLM.[/bold red]")
                break

            for block in response.content:
                if block.type == "text":
                    final_text_parts.append(block.text)
                    assistant_response_content_blocks.append(
                        {"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_calls_to_make.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                    assistant_response_content_blocks.append(
                        block.model_dump())

            if tool_calls_to_make:
                conversation_history.append(
                    {"role": "assistant", "content": assistant_response_content_blocks})
                tool_result_messages = []

                for tool_call in tool_calls_to_make:
                    tool_name = tool_call["name"]
                    tool_input = tool_call["input"]
                    tool_use_id = tool_call["id"]

                    console.print(
                        f"[bold yellow]🔧 Calling tool '{tool_name}' with input:[/bold yellow] {tool_input}")
                    text_output = None
                    tool_output_content = None
                    is_error_flag = False

                    try:
                        if not isinstance(tool_input, dict):
                            raise ValueError(
                                "Tool input must be a dictionary.")

                        input_schema = tool_map.get(
                            tool_name, {}).get("input_schema")
                        if input_schema:
                            validate(instance=tool_input, schema=input_schema)

                        mcp_tool_result = await session.call_tool(tool_name, arguments=tool_input)
                        content = getattr(mcp_tool_result, 'content', None)
                        if content and isinstance(content, list) and hasattr(content[0], 'text'):
                            text_output = content[0].text
                            try:
                                tool_output_content = json.loads(text_output)
                            except json.JSONDecodeError:
                                tool_output_content = {
                                    "error": "Non-JSON output from tool."}
                                is_error_flag = True
                        else:
                            tool_output_content = {
                                "error": "Tool returned no usable content."}
                            is_error_flag = True
                    except Exception as e:
                        logger.error("Tool call failed: %s", e)
                        tool_output_content = {"error": str(e)}
                        is_error_flag = True

                    if is_error_flag:
                        result_block = {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f"Tool execution failed: {tool_output_content.get('error', 'Unknown error')}"
                        }
                    else:
                        json_str = json.dumps(tool_output_content, indent=2)
                        console.print(
                            Syntax(json_str, "json", theme="monokai", line_numbers=True))
                        result_block = {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json_str
                        }

                    tool_result_messages.append(result_block)

                conversation_history.append(
                    {"role": "user", "content": tool_result_messages})
                continue
            else:
                if final_text_parts:
                    console.print(
                        f"[bold green]🤖 Claude:[/bold green] {' '.join(final_text_parts)}")
                    conversation_history.append(
                        {"role": "assistant", "content": assistant_response_content_blocks})
                break


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_config", default="llm_config.json")
    parser.add_argument("--mcp_config", default="mcp_servers.json")
    parser.add_argument(
        "--file", help="Path to a file (PDF, image, etc.) to attach to the prompt.")
    args = parser.parse_args()

    try:
        with open(args.llm_config, "r") as f:
            raw_llm_config = json.load(f)
            llm_config = TypeAdapter(LLMConfig).validate_python(raw_llm_config)

        with open(args.mcp_config, "r") as f:
            mcp_config = json.load(f)
    except Exception as e:
        console.print(
            f"[bold red]❌ Failed to load config files:[/bold red] {e}")
        return

    mcp_servers = mcp_config.get("mcpServers", {})
    for server_name, server_data in mcp_servers.items():
        command = server_data.get("command")
        args_list = server_data.get("args", [])
        env = server_data.get("env", {})

        try:
            params = StdioServerParameters(
                command=command, args=args_list, env=env)
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await run_session(session, llm_config, args)
                    break
        except Exception as e:
            console.print(
                f"[bold red]❌ Failed to start session with '{server_name}':[/bold red] {e}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        anyio.run(main)
    except* Exception as eg:
        console.print(
            "[bold red]\n🚨 Unhandled exception(s) occurred:[/bold red]")
        for ex in eg.exceptions:
            console.print(f"[red]- {type(ex).__name__}: {ex}[/red]")
