import argparse
import json
import logging
import anyio
import anthropic
# Assuming these base classes exist
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# Make sure necessary types are imported if they exist in your library, e.g.:
# from mcp.types import Tool, ListToolsResult, CallToolResult, ContentBlock

# --- Logging Setup ---
# Increased logging level to capture DEBUG messages added for diagnosis
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all added logs
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# Optionally, reduce verbosity of libraries if needed
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("enhanced_mcp_client")

# --- Configuration Loading ---


def load_configurations(llm_config_file: str, mcp_config_file: str):
    """Loads LLM and MCP server configurations from JSON files."""
    llm_config = None
    mcp_servers = {}

    # Load LLM Config
    try:
        with open(llm_config_file, "r") as f:
            llm_config = json.load(f)
        logger.info("Loaded LLM configuration from %s", llm_config_file)
        if not llm_config or "api_key" not in llm_config:
            logger.error("LLM configuration is missing or lacks 'api_key'.")
            llm_config = None  # Invalidate config
    except FileNotFoundError:
        logger.error("LLM configuration file not found: %s", llm_config_file)
    except json.JSONDecodeError as e:
        logger.error(
            "Error decoding LLM configuration JSON from %s: %s", llm_config_file, e)
    except Exception as e:
        logger.error("Unexpected error loading LLM configuration: %s", e)

    # Load MCP Config
    try:
        with open(mcp_config_file, "r") as f:
            mcp_config_data = json.load(f)
        logger.info("Loaded MCP server configuration from %s", mcp_config_file)
        if not isinstance(mcp_config_data.get("mcpServers"), dict):
            logger.error(
                "Missing or malformed 'mcpServers' section in MCP configuration."
            )
        else:
            mcp_servers = mcp_config_data["mcpServers"]
    except FileNotFoundError:
        logger.error(
            "MCP server configuration file not found: %s", mcp_config_file)
    except json.JSONDecodeError as e:
        logger.error(
            "Error decoding MCP server configuration JSON from %s: %s", mcp_config_file, e)
    except Exception as e:
        logger.error(
            "Unexpected error loading or parsing MCP server configuration: %s", e
        )

    return llm_config, mcp_servers

# --- LLM Interaction (with enhanced logging) ---


async def query_llm(client: anthropic.AsyncAnthropic, model: str, messages: list, processed_tools: list):
    """Sends messages to the Anthropic API and returns the response."""
    logger.info("--- Sending Request to LLM ---")
    logger.info(f"Model: {model}")
    # Log the full messages structure right before sending
    # Use DEBUG level for potentially large payload
    logger.debug("Messages Payload:")
    try:
        # Use json.dumps for pretty printing the potentially complex messages structure
        messages_json = json.dumps(messages, indent=2)
        logger.debug(messages_json)
    except Exception as log_e:
        logger.error(f"Could not serialize messages for logging: {log_e}")
        # Log raw if serialization fails
        logger.debug(f"Messages raw (might be large/complex): {messages}")

    logger.debug("Tools Payload:")  # Log tools as well
    try:
        tools_json = json.dumps(processed_tools, indent=2)
        logger.debug(tools_json)
    except Exception as log_e:
        logger.error(f"Could not serialize tools for logging: {log_e}")
        logger.debug(f"Tools raw: {processed_tools}")
    logger.info("--- End LLM Request ---")

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            messages=messages,
            tools=processed_tools
        )
        # Log successful response at DEBUG
        logger.debug("LLM Raw Response: %s", response)
        return response
    except anthropic.BadRequestError as e:
        logger.error("--- LLM BadRequestError ---")
        logger.error(f"Status Code: {e.status_code}")
        # Log the detailed error body provided by Anthropic
        logger.error(f"Error Body: {e.body}")
        # Log the request that failed again, for direct comparison
        logger.error("Messages Payload that caused error:")
        try:
            messages_json_error = json.dumps(messages, indent=2)
            logger.error(messages_json_error)
        except Exception as log_e:
            logger.error(
                f"Could not serialize messages for error logging: {log_e}")
            logger.error(f"Messages raw: {messages}")
        logger.error("--- End LLM BadRequestError ---")
        raise e  # Re-raise the exception to be caught by the outer handler in run_session
    except anthropic.APIConnectionError as e:
        logger.exception("LLM API Connection Error: %s", e)
        raise  # Re-raise
    except anthropic.RateLimitError as e:
        logger.exception("LLM Rate Limit Error: %s", e)
        raise  # Re-raise
    except anthropic.AuthenticationError as e:
        logger.exception("LLM Authentication Error: %s", e)
        raise  # Re-raise
    except Exception as e:  # Catch other potential API errors
        logger.exception(
            "An unexpected error occurred during LLM API call: %s", e)
        raise  # Re-raise


# --- User Input ---
async def get_user_input():
    """Gets user input asynchronously."""
    # anyio.to_thread.run_sync handles blocking input in async context
    return await anyio.to_thread.run_sync(input)

# --- Main Session Logic (with enhanced logging) ---


async def run_session(session: ClientSession, llm_config: dict):
    """Runs the main chat loop, interacting with MCP and LLM."""
    try:
        logger.info("Initializing MCP session...")
        await session.initialize()

        logger.info("Listing tools from MCP server...")
        # Assuming ListToolsResult has a 'tools' attribute which is a list
        results = await session.list_tools()

        if not results or not hasattr(results, 'tools') or not results.tools:
            logger.error(
                "No tools found on the MCP server or results format unexpected. Cannot proceed.")
            return

        # Prepare tools for Claude API format
        processed_tools = []
        tool_map = {}  # For easy lookup later
        logger.debug("Raw tools listed: %s", results.tools)
        for tool in results.tools:
            # Adjust based on actual 'tool' object structure and serialization method
            try:
                # Use model_dump() for Pydantic v2+, dict() for v1
                if hasattr(tool, 'model_dump'):
                    raw_tool = tool.model_dump()
                elif hasattr(tool, 'dict'):
                    raw_tool = tool.dict()
                else:
                    logger.error(
                        "Cannot serialize tool object: %s. Lacks .model_dump() or .dict()", tool)
                    continue

                tool_name = raw_tool.get('name', 'UNKNOWN_TOOL')
                # Get and remove original schema key
                input_schema = raw_tool.pop("inputSchema", None)

                # Check if schema exists and is a dict
                if isinstance(input_schema, dict) and input_schema:
                    # Add back with correct key name
                    raw_tool["input_schema"] = input_schema
                    processed_tools.append(raw_tool)
                    # Store for potential validation later
                    tool_map[tool_name] = raw_tool
                    logger.debug("Processed tool '%s' for LLM.", tool_name)
                else:
                    logger.warning(
                        "Tool '%s' skipped: Missing or invalid 'inputSchema'. Schema was: %s", tool_name, input_schema)
            except AttributeError as e:
                logger.error("Error processing tool object %s: %s.",
                             tool, e, exc_info=True)
            except Exception as e:
                logger.error(
                    "Unexpected error processing tool %s: %s", tool, e, exc_info=True)

        if not processed_tools:
            logger.error(
                "No valid tools could be processed for the LLM. Exiting session.")
            return

        logger.info("Tools prepared for LLM: %s", [
                    t['name'] for t in processed_tools])

        # Initialize Anthropic client
        try:
            client = anthropic.AsyncAnthropic(api_key=llm_config["api_key"])
            model = llm_config.get(
                "model", "claude-3-haiku-20240307")  # Default model
            logger.info("Anthropic client initialized for model: %s", model)
        except Exception as e:
            logger.exception("Failed to initialize Anthropic client: %s", e)
            print(
                f"🚨 Error: Could not initialize LLM client. Check API key and configuration. {e}")
            return

        conversation_history = []  # Stores the turn-by-turn history

        print("\n💬 Chat started! Type 'exit' or 'quit' to leave.\n")

        while True:
            print("You: ", end="", flush=True)
            user_input_raw = await get_user_input()
            user_query = user_input_raw.strip()  # Strip whitespace

            if user_query.lower() in {"exit", "quit"}:
                print("\n👋 Exiting chat.")
                break

            if not user_query:  # Skip empty input
                continue

            # Add user message to the history list that will be sent to the API
            conversation_history.append(
                {"role": "user", "content": user_query})

            # --- LLM Interaction Loop ---
            while True:  # Inner loop to handle potential tool calls
                # Prepare messages for the current API call from history
                messages_to_send = conversation_history  # Send the whole history

                try:
                    logger.info("Querying LLM (turn %d)...",
                                len(messages_to_send))
                    response = await query_llm(client, model, messages_to_send, processed_tools)

                except anthropic.BadRequestError as e:
                    # Error logged in query_llm, print user message here
                    print(
                        f"🚨 Error: Invalid request sent to LLM. Check logs for details. {e}")
                    # Break inner loop, potentially allowing user to rephrase or fix context
                    break
                except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.AuthenticationError) as e:
                    # Specific API errors logged in query_llm
                    print(f"🚨 Error: LLM API issue. Check logs. {e}")
                    # Decide how to handle: break outer loop, retry, etc.
                    if isinstance(e, anthropic.AuthenticationError):
                        return  # Fatal
                    break  # Break inner loop for others
                except Exception as e:  # Catch other potential API or processing errors
                    logger.exception(
                        "An unexpected error occurred during LLM query or response processing: %s", e)
                    print(f"🚨 An unexpected error occurred: {e}")
                    break  # Break inner loop

                # Process the response
                # Raw content blocks for the assistant's turn in history
                assistant_response_content_blocks = []
                tool_calls_to_make = []  # Collect tool calls requested by the model
                final_text_parts = []  # Collect text parts from the response

                if not response or not response.content:
                    logger.warning("Received empty or null content from LLM.")
                    print("🤖 Claude: (Received empty response)")
                    break  # Break inner loop, user can try again

                logger.debug("Processing LLM response content blocks...")
                # Iterate through content blocks (text or tool_use)
                for i, block in enumerate(response.content):
                    logger.debug(f"Response block {i}: type={block.type}")
                    if block.type == "text":
                        # Log snippet
                        logger.debug(
                            f"  Text content: '{block.text[:100]}...'")
                        final_text_parts.append(block.text)
                        assistant_response_content_blocks.append(
                            {"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        logger.debug(
                            f"  Tool use: id={block.id}, name={block.name}, input={block.input}")
                        tool_calls_to_make.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                        # Add raw tool_use block Pydantic object/dict to history content
                        if hasattr(block, 'model_dump'):
                            assistant_response_content_blocks.append(
                                block.model_dump())
                        else:
                            assistant_response_content_blocks.append(
                                block.dict())  # Fallback for Pydantic v1
                    else:
                        logger.warning(
                            "Received unknown block type from LLM: %s", block.type)

                # --- Handle Tool Calls (if any) ---
                if tool_calls_to_make:
                    # Add the assistant's turn (requesting the tool) to conversation history *before* adding tool results
                    if assistant_response_content_blocks:
                        logger.debug(
                            "Appending assistant message (with tool requests) to history.")
                        conversation_history.append({
                            "role": "assistant",
                            "content": assistant_response_content_blocks
                        })

                    logger.info(
                        f"LLM requested {len(tool_calls_to_make)} tool call(s).")
                    # Content blocks for the *next* user message reporting results
                    tool_result_messages = []

                    for tool_call in tool_calls_to_make:
                        tool_name = tool_call["name"]
                        tool_input = tool_call["input"]
                        tool_use_id = tool_call["id"]

                        logger.info(
                            "Calling tool '%s' via MCP with input: %s", tool_name, tool_input)
                        text_output = None  # Initialize
                        tool_output_content = None  # Initialize
                        is_error_flag = False  # Initialize

                        try:
                            # Call the actual tool via MCP session
                            mcp_tool_result = await session.call_tool(tool_name, arguments=tool_input)
                            logger.debug(
                                "MCP Tool Raw Result for '%s': %s", tool_name, mcp_tool_result)

                            # Process the result (assuming CallToolResult structure)
                            # Adapt this based on the actual structure of mcp_tool_result
                            if not mcp_tool_result or not hasattr(mcp_tool_result, 'content') or \
                               not mcp_tool_result.content or len(mcp_tool_result.content) == 0 or \
                               not hasattr(mcp_tool_result.content[0], 'text'):
                                logger.error(
                                    f"Tool '{tool_name}' returned unexpected/empty MCP content struct: {mcp_tool_result}")
                                tool_output_content = {
                                    "error": "Tool returned empty or unexpected content structure."}
                                is_error_flag = True
                            else:
                                text_output = mcp_tool_result.content[0].text
                                logger.debug(
                                    "Tool '%s' Raw Text Output: <<< %s >>>", tool_name, text_output)

                                try:
                                    # Attempt to parse as JSON - THIS IS THE KEY
                                    parsed_output = json.loads(text_output)
                                    tool_output_content = parsed_output  # Use the parsed dictionary
                                    is_error_flag = False
                                    logger.info(
                                        "Tool '%s' output successfully parsed as JSON.", tool_name)
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Tool '{tool_name}' output was not valid JSON. Reporting structured error. Raw: '{text_output}'")
                                    # Report a structured error back to the LLM
                                    tool_output_content = {
                                        "error": "Tool execution resulted in non-JSON output.",
                                        # "raw_output": text_output # Optional: include raw output if helpful and safe
                                    }
                                    is_error_flag = True

                        except Exception as e:
                            # Catch errors during MCP call or result processing
                            logger.exception(
                                "Error calling or processing MCP tool '%s': %s", tool_name, e)
                            tool_output_content = {
                                "error": f"Failed to execute or process tool '{tool_name}': {str(e)}"}
                            is_error_flag = True

                        # Construct the tool_result message content block for Anthropic
                        tool_result_block = {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "output": tool_output_content,  # Contains parsed JSON dict OR error dict
                        }
                        if is_error_flag:
                            tool_result_block["is_error"] = True

                        # +++ DETAILED LOGGING OF THE BLOCK BEING PREPARED +++
                        logger.debug(
                            "-----------------------------------------")
                        logger.debug("Preparing Tool Result Block for LLM:")
                        logger.debug(f"  Tool Name: {tool_name}")
                        logger.debug(f"  Tool Use ID: {tool_use_id}")
                        logger.debug(f"  Is Error Flag: {is_error_flag}")
                        logger.debug(
                            f"  Output Content Type: {type(tool_output_content)}")
                        try:
                            # Try logging as pretty JSON if it's dict/list, else log raw
                            output_log_str = json.dumps(
                                tool_output_content, indent=2)
                        except TypeError:
                            # Use repr for non-serializable types
                            output_log_str = repr(tool_output_content)
                        logger.debug(
                            f"  Output Content Value:\n{output_log_str}")
                        try:
                            # Log the final block as JSON
                            block_log_str = json.dumps(
                                tool_result_block, indent=2)
                        except TypeError:
                            block_log_str = repr(tool_result_block)
                        logger.debug(
                            f"  Complete tool_result_block:\n{block_log_str}")
                        logger.debug(
                            "-----------------------------------------")
                        # +++ END OF DETAILED LOGGING +++

                        # Add this tool result block to the list for the next user message
                        tool_result_messages.append(tool_result_block)
                        # --- End loop for single tool call ---

                    # Add a single user message containing *all* the tool results back to the history
                    if tool_result_messages:
                        user_tool_response_message = {
                            "role": "user",
                            "content": tool_result_messages  # List of tool result blocks
                        }
                        # +++ Log the message being added to history +++
                        logger.debug(
                            "Adding User message with tool results to history:")
                        try:
                            logger.debug(
                                f"{json.dumps(user_tool_response_message, indent=2)}")
                        except TypeError:
                            logger.debug(f"{repr(user_tool_response_message)}")
                        # +++ End log +++
                        conversation_history.append(user_tool_response_message)

                    # Continue the inner loop to send the tool results back to the LLM
                    logger.info("Re-querying LLM with tool results.")
                    continue  # Go back to query_llm with the updated history

                # --- Handle Text Response (if no tool calls were made in *this* response) ---
                else:
                    logger.debug("No tool calls requested in this response.")
                    final_text = "\n".join(final_text_parts).strip()
                    if final_text:
                        print(f"🤖 Claude: {final_text}")
                        # Add the final assistant text response to conversation history
                        if assistant_response_content_blocks:
                            logger.debug(
                                "Appending final assistant text message to history.")
                            conversation_history.append({
                                "role": "assistant",
                                "content": assistant_response_content_blocks  # Use the structured content
                            })
                    else:
                        # This might happen if the LLM responds only with tool use requests
                        # which were handled above, or if the response was truly empty.
                        logger.info(
                            "LLM response did not contain final text after processing.")
                        # Avoid adding an empty assistant message if there was no text and no prior tool use block added
                        if not assistant_response_content_blocks and conversation_history[-1]['role'] != 'assistant':
                            print("🤖 Claude: (No text content in response)")

                    # Break the inner loop (no more tool calls expected for this user query turn)
                    logger.debug("Breaking inner LLM loop.")
                    break

            # --- End of Inner LLM Interaction Loop ---
            logger.debug("Exited inner LLM loop.")

    except ConnectionRefusedError as e:
        logger.error(
            "Connection refused when trying to initialize MCP session: %s", e)
        print(
            f"🚨 Error: Could not connect to the MCP server process. Is it running? {e}")
    except FileNotFoundError as e:
        logger.error("MCP server command not found: %s", e)
        print(
            f"🚨 Error: The command to start the MCP server was not found. Check config. {e}")
    except Exception as e:
        # Catch errors during session initialization or the outer loop
        logger.exception("Unhandled error during session run: %s", e)
        print(f"⚠️ An unexpected session error occurred: {e}")
    finally:
        logger.info("MCP session finished or encountered an error.")
        # Cleanup logic for the session could go here if needed


# --- Main Execution ---
async def main():
    """Parses arguments, loads config, and starts the MCP client session."""
    parser = argparse.ArgumentParser(
        description="Enhanced MCP Terminal Client with Anthropic LLM integration"
    )
    parser.add_argument(
        "--llm_config", default="llm_config.json",
        help="Path to LLM configuration file (default: llm_config.json)"
    )
    parser.add_argument(
        "--mcp_config", default="mcp_servers.json",
        help="Path to MCP servers configuration file (default: mcp_servers.json)"
    )
    args = parser.parse_args()

    llm_config, mcp_servers = load_configurations(
        args.llm_config, args.mcp_config
    )

    if not llm_config:
        logger.critical(
            "LLM configuration loading failed or incomplete. Cannot start.")
        print("🚨 Error: LLM configuration failed. Check llm_config.json and logs.")
        return
    if not mcp_servers:
        logger.critical(
            "MCP server configuration loading failed or empty. Cannot start.")
        print("🚨 Error: MCP server configuration failed. Check mcp_servers.json and logs.")
        return

    # Attempt to connect to the first configured server found
    connected_server_name = None
    for server_name, server_config in mcp_servers.items():
        command = server_config.get("command")
        if not command:
            logger.warning(
                "Skipping server '%s': missing 'command' in configuration.", server_name
            )
            continue

        logger.info("Attempting to connect to MCP server '%s'...", server_name)
        try:
            params = StdioServerParameters(
                command=command,
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )
            logger.debug("stdio params: command=%s, args=%s",
                         command, server_config.get("args", []))

            # Use stdio_client context manager for reliable process management
            async with stdio_client(params) as (read_stream, write_stream):
                logger.info(
                    "stdio streams established for '%s'. Initializing ClientSession...", server_name)
                async with ClientSession(read_stream, write_stream) as session:
                    logger.info(
                        "MCP ClientSession connected to server '%s'. Starting chat run...", server_name)
                    connected_server_name = server_name
                    await run_session(session, llm_config)
                    # Session completed normally for this server
                    break  # Exit loop after successful connection and session run

        except ConnectionRefusedError as e:
            logger.warning(
                "Connection refused for server '%s': %s. Ensure server process can start.", server_name, e)
        except FileNotFoundError as e:
            logger.warning("Command not found for server '%s': '%s'. Check command path.",
                           server_name, command, exc_info=True)
        except Exception as e:
            # Catch broader errors during connection/session setup
            logger.warning(
                # Log stack trace
                "Failed to connect or run session with server '%s': %s.", server_name, e, exc_info=True
            )
        # Automatically cleans up stdio_client and ClientSession contexts here

    if not connected_server_name:
        logger.error(
            "Failed to connect to any configured MCP server. Exiting.")
        print("\n🚨 Error: Could not connect to any MCP server defined in mcp_servers.json. Check server configurations and logs.")
    else:
        logger.info("Finished session with server '%s'.",
                    connected_server_name)


if __name__ == "__main__":
    try:
        # Set high-level logger for anyio if it's too noisy
        # logging.getLogger('anyio').setLevel(logging.WARNING)
        anyio.run(main)
    except KeyboardInterrupt:
        logger.info("Client terminated by user (KeyboardInterrupt).")
        print("\n👋 Client terminated.")
    except Exception as e:
        # Catch any unexpected errors from the top level
        logger.critical("Fatal error in main execution: %s", e, exc_info=True)
        print(f"\n🚨 A critical error occurred: {e}")
