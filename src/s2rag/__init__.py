__all__ = ["S2RAG", "BaseConfig", "build_s2rag_config"]


def __getattr__(name):
    if name == "S2RAG":
        from .S2RAG import S2RAG

        return S2RAG
    if name == "BaseConfig":
        from .utils.config_utils import BaseConfig

        return BaseConfig
    if name == "build_s2rag_config":
        from .profile import build_s2rag_config

        return build_s2rag_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
