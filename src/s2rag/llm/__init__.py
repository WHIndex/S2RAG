from ..utils.config_utils import BaseConfig
from .compatible_chat import CachedChatClient
from .base import BaseLLM


def _get_llm_class(config: BaseConfig):
    return CachedChatClient.from_experiment_config(config)
