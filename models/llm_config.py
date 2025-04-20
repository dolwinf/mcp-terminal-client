from pydantic import BaseModel
from typing import Optional, Literal, Union


class BaseLLMConfig(BaseModel):
    provider: str
    model: str


class AnthropicConfig(BaseLLMConfig):
    provider: Literal["anthropic"]
    model: str


class VertexAnthropicConfig(BaseLLMConfig):
    provider: Literal["vertexAnthropic"]
    model: str
    project_id: str
    region: str


LLMConfig = Union[AnthropicConfig, VertexAnthropicConfig]
