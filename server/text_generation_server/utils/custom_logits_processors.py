from typing import Dict, Iterable
from transformers import PreTrainedTokenizerBase, LogitsWarper
import abc


class CustomLogitsWarperFactory(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def create_warper(self, tokenizer: PreTrainedTokenizerBase, params: Iterable[str]) -> LogitsWarper:
        raise NotImplementedError


class CustomLogitsProcessorsManager:
    processors: Dict[str, CustomLogitsWarperFactory] = {}
    
    @classmethod
    def register_factory(cls, factory: CustomLogitsWarperFactory):
        """Register a factory for a custom warper. This should be called by library developers in the __init__.py of their custom warper module,
        and the module should be included in the custom_modules command line argument of the text-generation-server CLI."""
        cls.processors[factory.name] = factory

    @classmethod
    def create_warper(cls, name: str, tokenizer: PreTrainedTokenizerBase, params: Iterable[str]) -> LogitsWarper:
        """Create a custom warper by name."""
        if name not in cls.processors:
            raise ValueError(f"Unknown warper {name}. Known warpers: {', '.join(cls.processors.keys())}")
        return cls.processors[name].create_warper(tokenizer, params)
