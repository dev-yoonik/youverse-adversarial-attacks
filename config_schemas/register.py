from .attack_schemas import register_config as register_attack_schemas
from .model_schemas import register_config as register_model_schemas
from .dataset_attacker_schemas import register_config as register_dataset_attacker_schemas


def register_config() -> None:
    register_attack_schemas()
    register_model_schemas()
    register_dataset_attacker_schemas()
