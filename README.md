# MCP Terminal Client

A client for interacting with Model Control Protocol (MCP) servers, integrated with the Anthropic Claude API for enhanced LLM capabilities.

## Overview

This client provides an interface between MCP tool servers and Anthropic's Claude language models, allowing for:

- Interactive command-line chat with Claude
- Dynamic tool discovery from MCP servers
- Automatic tool execution via Claude's function calling capabilities
- Detailed logging

## Requirements

- Python 3.11+
- `anthropic` Python package
- `anyio` Python package
- `mcp` Python package

## Configuration

The client requires two configuration files:

### 1. `llm_config.json`

Contains Anthropic API credentials and model selection:

```json
{
  "api_key": "your_anthropic_api_key",
  "model": "claude-3-haiku-20240307"
}
```

### 2. `mcp_servers.json`

Defines available MCP servers:

```json
{
  "mcpServers": {
    "default": {
      "command": "/path/to/mcp_server",
      "args": ["--option1", "value1"],
      "env": {
        "ENV_VAR1": "value1"
      }
    },
    "alternative": {
      "command": "/path/to/another_server",
      "args": [],
      "env": {}
    }
  }
}
```

## Usage

Run the client with default configuration files:

```bash
python enhanced_mcp_client.py
```

Specify custom configuration paths:

```bash
python enhanced_mcp_client.py --llm_config custom_llm_config.json --mcp_config custom_mcp_servers.json
```

## Features

### Tool Integration

- Automatic discovery of available tools from MCP servers
- JSON schema validation for tool inputs
- Structured handling of tool results

## Conversation Flow

1. User enters a query
2. Query is sent to Claude with available tools
3. Claude responds with text and/or tool use requests
4. Tool requests are executed via MCP server
5. Tool results are sent back to Claude
6. Claude provides final response

## Troubleshooting

If you encounter issues:

1. Check the log output (set to DEBUG level by default)
2. Verify MCP server is running and accessible
3. Confirm your Anthropic API key is valid
4. Inspect tool configurations in the MCP server


# TODO

1. Support for VertexAI, OpenAI
2. Organise code to make the client vendor agnostic
3. Add ruff
4. Add test cases
5. Add SSE support