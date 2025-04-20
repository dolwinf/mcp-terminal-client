import os
import anthropic
from anthropic import AsyncAnthropicVertex, APIError
import logging

logger = logging.getLogger(__name__)


class LLMProvider:
    def __init__(self, config: dict):
        """
        Initializes the LLM provider.

        Args:
            config: A dictionary containing configuration for the provider,
                    including 'model' and potentially other provider-specific keys.
        """
        self.config = config
        self.model = config.get("model")
        if not self.model:
            raise ValueError("LLM configuration must include a 'model'.")

    async def send_message(self, messages: list, tools: list | None = None):
        """
        Sends a message list to the LLM and returns the response.

        Args:
            messages: A list of message dictionaries representing the conversation history.
            tools: An optional list of tool definitions to provide to the LLM.

        Returns:
            The raw response object from the LLM API.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            APIError: If the API call fails.
            Exception: For other potential errors.
        """
        raise NotImplementedError(
            "send_message must be implemented by subclasses")


class AnthropicProvider(LLMProvider):
    def __init__(self, config: dict):
        super().__init__(config)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing ANTHROPIC_API_KEY environment variable for AnthropicProvider.")
        # Initialize the client once during instantiation
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info(f"AnthropicProvider initialized with model: {self.model}")

    async def send_message(self, messages: list, tools: list | None = None):
        logger.debug(
            f"Sending message to Anthropic model {self.model} with {len(messages)} history messages and {len(tools) if tools else 0} tools.")
        try:
            # Note: Anthropic API expects 'messages', not 'conversation'
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,  # Revisit to make this dynamic
                messages=messages,
                tools=tools if tools else []  # Pass tools if provided
            )
            logger.debug(
                f"Received response from Anthropic. Stop reason: {response.stop_reason}")

            return response
        except APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during Anthropic API call: {e}")
            raise


class VertexAnthropicProvider(LLMProvider):
    def __init__(self, config: dict):
        super().__init__(config)
        self.project_id = self.config.get("project_id")
        self.region = self.config.get("region")
        if not self.project_id or not self.region:
            raise ValueError(
                "Missing project_id or region in config for VertexAnthropicProvider.")
        # Initialize the client once during instantiation
        self.client = AsyncAnthropicVertex(
            project_id=self.project_id, region=self.region)
        logger.info(
            f"VertexAnthropicProvider initialized with model: {self.model}, project: {self.project_id}, region: {self.region}")

    async def send_message(self, messages: list, tools: list | None = None):
        logger.debug(
            f"Sending message to Vertex Anthropic model {self.model} with {len(messages)} history messages and {len(tools) if tools else 0} tools.")
        try:
            # Note: Vertex Anthropic API expects 'messages', not 'conversation'
            response = await self.client.messages.create(
                # The model name here might need adjustment for Vertex (e.g., claude-3-sonnet@20240229)
                model=self.model,
                max_tokens=4096,  # Consider making this configurable
                messages=messages,
                tools=tools if tools else []
            )
            logger.debug(
                f"Received response from Vertex Anthropic. Stop reason: {response.stop_reason}")
            t
            return response
        except APIError as e:
            logger.error(f"Vertex Anthropic API error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during Vertex Anthropic API call: {e}")
            raise


def get_llm_provider(config: dict) -> LLMProvider:
    """
    Function to create an LLMProvider instance based on the config.
    """
    provider_name = config.get("provider")
    if provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name == "vertexAnthropic":
        return VertexAnthropicProvider(config)
    # more blocks here for future providers (e.g., openai, gemini)

    else:
        raise ValueError(f"Unsupported LLM provider type: {provider_name}")
