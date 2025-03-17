from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from typing import Tuple, Any, Optional
from omegaconf import MISSING


@dataclass
class BaseModelConfig:
    """
    Base class for all model schemas
    """


@dataclass
class InsightFaceFRModelConfig(BaseModelConfig):
    """
    Configuration for InsightFaceFRModel
    :param model_path: path to the model
    :param dropout_model: whether to use dropout model
    :param device: device to perform the attack on
    """
    model_type: str = 'insightface'
    model_path: str = MISSING
    dropout_model: bool = False
    device: str = 'cuda:0'

@dataclass
class FaceNetModelConfig(BaseModelConfig):
    """
    Configuration for FaceNetModel
    :param pretrained_type: type of pretrained model ['vggface2', 'casia-webface']
    :param device: device to perform the attack on
    """
    model_type: str = 'facenet'
    pretrained_type: str = 'casia-webface'
    device: str = 'cuda:0'


@dataclass
class EnsembleModelConfig(BaseModelConfig):
    """
    Configuration for EnsembleModel
    :param models: list of models to ensemble
    """
    model_type: str = 'ensemble'


def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="schemas/models", name="base", node=BaseModelConfig)
    cs.store(group="schemas/models", name="insightface", node=InsightFaceFRModelConfig)
    cs.store(group="schemas/models", name="facenet", node=FaceNetModelConfig)
    cs.store(group="schemas/models", name="ensemble", node=EnsembleModelConfig)