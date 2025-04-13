import argparse
import json
import logging
import anyio
import os
import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from jsonschema import validate, ValidationError
from rich.console import Console
from rich.syntax import Syntax
from rich import print as rprint
import traceback

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("python_mcp_client")
console = Console()


async def run_session(session: ClientSession, llm_config: dict):
    console.print(
        "[bold cyan]\n💬 Chat started! Type 'exit' or 'quit' to leave.\n[/bold cyan]")

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

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[bold red]❌ Missing ANTHROPIC_API_KEY environment variable.[/bold red]")
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    model = llm_config.get("model", "claude-3-haiku-20240307")
    conversation_history = []

    while True:
        # Using raw print to avoid Rich error
        print("\033[1;35mYou:\033[0m ", end="", flush=True)
        user_input_raw = await anyio.to_thread.run_sync(input)
        user_query = user_input_raw.strip()

        if user_query.lower() in {"exit", "quit"}:
            console.print("\n[bold cyan]👋 Exiting chat.[/bold cyan]")
            break

        if not user_query:
            continue

        conversation_history.append({"role": "user", "content": user_query})

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
                            "content": f"Tool execution failed: {str(tool_output_content.get('error', 'Unknown error'))}"
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
                continue  # Re-query LLM with tool results
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
    args = parser.parse_args()

    try:
        with open(args.llm_config, "r") as f:
            llm_config = json.load(f)
        with open(args.mcp_config, "r") as f:
            mcp_config = json.load(f)
    except Exception as e:
        console.print(
            f"[bold red]❌ Failed to load config files:[/bold red] {e}")
        return

    mcp_servers = mcp_config.get("mcpServers", {})
    for server_name, server_data in mcp_servers.items():
        command = server_data.get("command")
        args = server_data.get("args", [])
        env = server_data.get("env", {})

        try:
            params = StdioServerParameters(command=command, args=args, env=env)
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await run_session(session, llm_config)
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
