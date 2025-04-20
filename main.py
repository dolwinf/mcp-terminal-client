import argparse
import json
import logging
import anyio
import os
import sys
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
from pydantic import BaseModel, model_validator, TypeAdapter, Field
from typing import Annotated, Literal, Union
from llm_provider import LLMProvider, get_llm_provider, APIError
from utilities.anthropic_utils import schema_processor

if os.name == "nt":
    # Revisit to check if logging is better than redirecting stderr
    try:
        sys.stderr = open(os.devnull, "w")
    except Exception:
        g
        print("Warning: Could not redirect stderr.", file=sys.__stderr__)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.INFO)


logger = logging.getLogger("mcp_terminal_client")
console = Console()

SUPPORTED_FILE_TYPES = {
    "application/pdf": "document",
    "image/png": "image",
    "image/jpeg": "image",
    "image/webp": "image",
    "image/gif": "image",
    "text/plain": "document",
    "text/markdown": "document",
}


class BaseLLMConfig(BaseModel):
    provider: str  # Keep as str initially for broader validation
    model: str
    # Add common optional parameters if needed
    # max_tokens: int = 4096


class AnthropicConfig(BaseLLMConfig):
    provider: Literal["anthropic"]


class VertexAnthropicConfig(BaseLLMConfig):
    provider: Literal["vertexAnthropic"]
    project_id: str
    region: str

# Add future provider configs here
# class OpenAIConfig(BaseLLMConfig):
#    provider: Literal["openai"]
#    # Add OpenAI specific fields if any


LLMConfig = Annotated[
    Union[
        AnthropicConfig,
        VertexAnthropicConfig,
        # OpenAIConfig, # Add future configs here
    ],
    Field(discriminator="provider"),
]

# --- Main Chat Logic ---


async def run_session(
    mcp_session: ClientSession,
    provider: LLMProvider,
    args: argparse.Namespace
):
    console.print(
        "\n[bold cyan]💬 Chat started! Type 'exit' or 'quit' to leave.[/bold cyan]")

    logger.info("Initializing MCP session...")
    await mcp_session.initialize()

    logger.info("Listing tools from MCP server...")
    try:
        results = await mcp_session.list_tools()
        processed_tools, tool_map = schema_processor(results)
    except Exception as e:
        logger.error(
            f"Failed to list tools from MCP server: {e}", exc_info=True)
        console.print(
            f"[bold red]❌ Error communicating with MCP server: {e}[/bold red]")
        return

    conversation_history = []

    # File Handling
    file_attachment_content = None
    if args.file:
        file_path = args.file
        if not os.path.isfile(file_path):
            console.print(
                f"[bold red]❌ File not found: {file_path}[/bold red]")
            return

        mime_type, _ = mimetypes.guess_type(file_path)
        file_type_category = SUPPORTED_FILE_TYPES.get(mime_type)

        if not file_type_category:
            console.print(
                f"[bold red]❌ Unsupported file type: {mime_type or 'unknown'}[/bold red]\n"
                f"Supported types: {', '.join(SUPPORTED_FILE_TYPES.keys())}"
            )
            return

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Prepare content block structure expected by Anthropic API
            file_attachment_content = {
                "type": file_type_category,
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64.b64encode(file_data).decode("utf-8"),
                }
            }
            console.print(
                f"[dim]📎 Attaching file: {os.path.basename(file_path)} ({mime_type})[/dim]")
        except Exception as e:
            logger.error(
                f"Error reading or encoding file {file_path}: {e}", exc_info=True)
            console.print(f"[bold red]❌ Error processing file: {e}[/bold red]")
            return

    file_attachment_used = False

    # Chat Loop
    while True:
        try:
            # Use anyio's input for async compatibility
            user_input_raw = await anyio.to_thread.run_sync(
                console.input, "[bold magenta]You:[/bold magenta] "
            )
            user_query = user_input_raw.strip()
        except (KeyboardInterrupt, EOFError):
            user_query = "exit"

        if user_query.lower() in {"exit", "quit"}:
            console.print("\n[bold cyan]👋 Exiting chat.[/bold cyan]")
            break

        if not user_query:
            continue

        # Process User Message
        user_content = []

        if file_attachment_content and not file_attachment_used:
            user_content.append(file_attachment_content)

            user_content.append({"type": "text", "text": user_query})
            file_attachment_used = True
            logger.info("Attached file to user message.")
        else:

            user_content.append({"type": "text", "text": user_query})

        conversation_history.append({"role": "user", "content": user_content})

        # Main LLM Interaction Loop (Handles potential tool calls)
        while True:
            try:
                logger.info(
                    f"Sending request to LLM (History: {len(conversation_history)} messages)")

                response = await provider.send_message(
                    messages=conversation_history,
                    tools=processed_tools
                )

            except APIError as e:
                console.print(f"[bold red]❌ LLM API Error:[/bold red] {e}")
                conversation_history.pop()
                break
            except Exception as e:
                logger.error(f"LLM request failed: {e}", exc_info=True)
                console.print(
                    f"[bold red]❌ LLM request failed unexpectedly:[/bold red] {e}")
                conversation_history.pop()
                break

            # Process LLM Response
            assistant_response_content_blocks = []
            tool_calls_to_make = []
            final_text_parts = []

            if not response or not response.content:
                console.print(
                    "[bold red]⚠️ Empty or invalid response from LLM.[/bold red]")

                break

            stop_reason = response.stop_reason
            logger.debug(f"LLM response received. Stop Reason: {stop_reason}")

            for block in response.content:
                if block.type == "text":
                    final_text_parts.append(block.text)
                    assistant_response_content_blocks.append(
                        {"type": "text", "text": block.text})
                elif block.type == "tool_use":

                    tool_use_id = getattr(block, 'id', None)
                    tool_name = getattr(block, 'name', None)
                    tool_input = getattr(block, 'input', None)

                    if not all([tool_use_id, tool_name, tool_input is not None]):
                        logger.warning(
                            f"Skipping malformed tool_use block: {block}")
                        continue

                    tool_calls_to_make.append({
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": tool_input
                    })

                    assistant_response_content_blocks.append(
                        block.model_dump())

            # Handle Tool Calls
            if tool_calls_to_make and stop_reason == "tool_use":

                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response_content_blocks
                })

                tool_result_messages = []

                # Execute tools
                for tool_call in tool_calls_to_make:
                    tool_name = tool_call["name"]
                    tool_input = tool_call["input"]
                    tool_use_id = tool_call["id"]

                    console.print(
                        f"\n[bold yellow]🔧 Calling tool '{tool_name}'...[/bold yellow]")

                    console.print("[dim]Input:[/dim]")
                    console.print(Syntax(json.dumps(tool_input, indent=2), lexer="json",
                                  theme="github-dark", line_numbers=False, word_wrap=True))

                    tool_output_content_str = ""
                    is_error_flag = False

                    try:

                        tool_info = tool_map.get(tool_name)
                        if not tool_info:
                            raise ValueError(
                                f"Tool '{tool_name}' not found or has no schema.")

                        input_schema = tool_info.get("input_schema")
                        if input_schema:

                            if not isinstance(tool_input, dict) and input_schema.get("type") == "object":
                                raise ValueError(
                                    "Tool input is not a dictionary, but schema expects one.")
                            validate(instance=tool_input, schema=input_schema)
                            logger.debug(
                                f"Tool input validation successful for '{tool_name}'.")
                        else:

                            logger.warning(
                                f"No input schema found for tool '{tool_name}' during call.")

                        # Call the MCP tool
                        mcp_tool_result = await mcp_session.call_tool(tool_name, arguments=tool_input)

                        # Process MCP tool result
                        content = getattr(mcp_tool_result, 'content', None)

                        raw_output = None
                        if content and isinstance(content, list) and content[0] and hasattr(content[0], 'text'):
                            raw_output = content[0].text
                            logger.debug(
                                f"Raw tool output for '{tool_name}': {raw_output}")

                            try:
                                tool_output_content_parsed = json.loads(
                                    raw_output)
                                tool_output_content_str = json.dumps(
                                    tool_output_content_parsed)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Tool '{tool_name}' output was not valid JSON. Treating as plain text.")

                                tool_output_content_str = raw_output
                        elif content:
                            logger.warning(
                                f"Tool '{tool_name}' returned unexpected content structure: {content}")
                            # Stringify for LLM
                            tool_output_content_str = f"Tool returned unexpected content: {str(content)}"
                            is_error_flag = True
                        else:
                            logger.warning(
                                f"Tool '{tool_name}' returned no content.")
                            tool_output_content_str = "Tool returned no content."
                            is_error_flag = True

                    except (validate.ValidationError, ValueError) as e:
                        logger.error(
                            f"Tool input validation failed for '{tool_name}': {e}")
                        tool_output_content_str = f"Error: Input validation failed - {e}"
                        is_error_flag = True
                    except Exception as e:
                        logger.error(
                            f"Tool call failed for '{tool_name}': {e}", exc_info=True)
                        tool_output_content_str = f"Error: Tool execution failed - {e}"
                        is_error_flag = True

                    # Prepare Tool Result Block for LLM
                    result_block = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_output_content_str,

                    }
                    tool_result_messages.append(result_block)

                    if is_error_flag:
                        console.print(
                            f"[bold red]❌ Tool Error:[/bold red] {tool_output_content_str}")
                    else:
                        console.print(
                            "[bold green]✅ Tool Result:[/bold green]")

                        try:
                            parsed_output = json.loads(tool_output_content_str)
                            console.print(Syntax(json.dumps(
                                parsed_output, indent=2), lexer="json", theme="github-dark", line_numbers=False, word_wrap=True))
                        except json.JSONDecodeError:
                            console.print(
                                f"[dim]{tool_output_content_str}[/dim]")

                conversation_history.append(
                    {"role": "user", "content": tool_result_messages})
                # Continue the inner loop to send results back to the LLM
                continue

            else:
                final_text = " ".join(final_text_parts).strip()
                if final_text:
                    console.print(
                        f"[bold green]🤖 Assistant:[/bold green] {final_text}")

                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_response_content_blocks  # Use the blocks received
                    })
                elif stop_reason != "tool_use":
                    console.print(
                        "[bold yellow]⚠️ LLM stopped without providing text or requesting tools.[/bold yellow]")

                break


async def main():
    parser = argparse.ArgumentParser(
        description="MCP Terminal Client with LLM Integration")
    parser.add_argument("--llm-config", default="llm_config.json",
                        help="Path to LLM configuration JSON file.")
    parser.add_argument("--mcp-config", default="mcp_servers.json",
                        help="Path to MCP server configuration JSON file.")
    parser.add_argument(
        "--file", help="Path to a file (PDF, image) to attach to the first prompt.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose debug logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")

    # Load Configurations
    try:
        logger.info(f"Loading LLM config from: {args.llm_config}")
        with open(args.llm_config, "r") as f:
            raw_llm_config = json.load(f)

            llm_config_validated = TypeAdapter(
                LLMConfig).validate_python(raw_llm_config)
            # Convert back to dict for provider instantiation
            llm_config_dict = llm_config_validated.model_dump()
            logger.info(
                f"LLM config loaded successfully for provider: {llm_config_dict.get('provider')}")

        logger.info(f"Loading MCP config from: {args.mcp_config}")
        with open(args.mcp_config, "r") as f:
            mcp_config = json.load(f)
            if "mcpServers" not in mcp_config:
                raise ValueError("MCP config missing 'mcpServers' key.")
            logger.info("MCP config loaded successfully.")

    except FileNotFoundError as e:
        logger.error(
            f"Configuration file not found: {e.filename}", exc_info=True)
        console.print(
            f"[bold red]❌ Configuration file not found: {e.filename}[/bold red]")
        return
    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(
            f"Failed to load or validate configuration files: {e}", exc_info=True)
        console.print(
            f"[bold red]❌ Failed to load or parse config files:[/bold red] {e}")
        return

    # Instantiate LLM Provider
    try:
        provider = get_llm_provider(llm_config_dict)
        logger.info(
            f"LLM Provider '{llm_config_dict['provider']}' instantiated.")
    except (ValueError, Exception) as e:
        logger.error(f"Failed to instantiate LLM provider: {e}", exc_info=True)
        console.print(
            f"[bold red]❌ Failed to set up LLM provider:[/bold red] {e}")
        return

    # Connect to MCP Server
    mcp_servers = mcp_config.get("mcpServers", {})
    connected = False
    for server_name, server_data in mcp_servers.items():
        command = server_data.get("command")
        args_list = server_data.get("args", [])
        env = server_data.get("env", {})

        if not command:
            logger.warning(
                f"Skipping server '{server_name}' due to missing 'command'.")
            continue

        logger.info(
            f"Attempting to connect to MCP server '{server_name}' using command: {' '.join([command] + args_list)}")
        try:
            params = StdioServerParameters(
                command=command, args=args_list, env=env)
            # Use asyncio.create_task for better async management, but stdio_client context manager handles it
            async with stdio_client(params) as (read_stream, write_stream):
                logger.info(
                    f"Connected to MCP server '{server_name}'. Creating client session.")
                async with ClientSession(read_stream, write_stream) as mcp_session:

                    await run_session(mcp_session, provider, args)
                    connected = True
                    break
        except ConnectionRefusedError:
            logger.error(
                f"Connection refused by MCP server '{server_name}'. Is it running?")
            console.print(
                f"[bold red]❌ Connection refused by MCP server '{server_name}'. Make sure it's running.[/bold red]")
        except FileNotFoundError:
            logger.error(
                f"Command not found for MCP server '{server_name}': {command}")
            console.print(
                f"[bold red]❌ Command not found for MCP server '{server_name}': {command}[/bold red]")
        except Exception as e:
            logger.error(
                f"Failed to start session with '{server_name}': {e}", exc_info=True)
            console.print(
                f"[bold red]❌ Failed to start session with '{server_name}':[/bold red] {e}")

    if not connected:
        console.print(
            "[bold red]❌ Could not connect to any configured MCP server.[/bold red]")


if __name__ == "__main__":
    try:
        anyio.run(main)
    except Exception as e:
        console.print(
            "[bold red]\n🚨 An unhandled exception occurred:[/bold red]")
        if isinstance(e, ExceptionGroup):
            for i, ex in enumerate(e.exceptions):
                console.print(
                    f"[red]- Exception {i+1}: {type(ex).__name__}: {ex}[/red]")
        else:
            console.print(f"[red]- {type(e).__name__}: {e}[/red]")
        console.print("[dim]Traceback:[/dim]")
        traceback.print_exc()

    finally:

        if os.name == "nt" and isinstance(sys.stderr, open) and sys.stderr.name == os.devnull:
            sys.stderr.close()
            sys.stderr = sys.__stderr__
