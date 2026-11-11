from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .base import EmbeddingConfig, BaseEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    raise ValueError(f"Unsupported embedding model: {embedding_model_name}")
