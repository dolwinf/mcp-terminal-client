from rich.console import Console
import logging

logger = logging.getLogger("mcp_terminal_client")

console = Console()


def schema_processor(results):

    processed_tools = []
    tool_map = {}

    if not results or not hasattr(results, 'tools') or not results.tools:
        console.print(
            "[bold yellow]⚠️ No tools found on the MCP server or results format unexpected.[/bold yellow]")

    else:

        for tool in results.tools:
            try:

                processed_tool = tool.model_dump(exclude_unset=True)
                tool_name = processed_tool.get('name')
                if not tool_name:
                    logger.warning(
                        "Skipping tool without a name: %s", processed_tool)
                    continue

                # Keep original case from MCP
                input_schema = processed_tool.get("inputSchema")
                # Ensure schema is a dict and not empty before adding
                if isinstance(input_schema, dict) and input_schema:
                    # Prepare tool for LLM (adjust if LLM expects different format)
                    anthropic_tool_schema = {
                        "name": tool_name,

                        "description": processed_tool.get("description", f"Tool named {tool_name}"),
                        "input_schema": input_schema
                    }
                    processed_tools.append(anthropic_tool_schema)

                    tool_map[tool_name] = {"input_schema": input_schema}
                    logger.info(f"Registered tool: {tool_name}")
                else:
                    logger.warning(
                        f"Skipping tool '{tool_name}' due to missing or invalid inputSchema.")

            except Exception as e:
                logger.warning(
                    f"Skipping tool due to processing error: {e}", exc_info=True)

    return processed_tools, tool_map
