import copy

from transformers import GPT2Config, ViTConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ViTGPT2Config(PretrainedConfig):

    model_type = "vit-gpt2"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "vit_config" not in kwargs:
            raise ValueError("`vit_config` can not be `None`.")

        if "gpt2_config" not in kwargs:
            raise ValueError("`gpt2_config` can not be `None`.")

        vit_config = kwargs.pop("vit_config")
        gpt2_config = kwargs.pop("gpt2_config")

        self.vit_config = ViTConfig(**vit_config)
        self.gpt2_config = GPT2Config(**gpt2_config)

    @classmethod
    def from_vit_gpt2_configs(
        cls, vit_config: PretrainedConfig, gpt2_config: PretrainedConfig, **kwargs
    ):
        return cls(
            vit_config=vit_config.to_dict(),
            gpt2_config=gpt2_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vit_config"] = self.vit_config.to_dict()
        output["gpt2_config"] = self.gpt2_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output