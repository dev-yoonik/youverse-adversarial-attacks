from hydra.utils import instantiate
from .attack_schemas import *
from .model_schemas import *
from .dataset_attacker_schemas import *
from fr_attacks.base_attack import BaseAttack
from models.base_model import BaseAttackModel


def instantiate_model(model_config: BaseModelConfig) -> BaseAttackModel:
    """
    Instantiates a surrogate model
    :param model_config: Config for the model
    :return: model object
    """
    if model_config.model_type == 'insightface':
        if not model_config.dropout_model:
            from models.model_insightface import load_insightface_ir50_model
            model = load_insightface_ir50_model(
                model_config.model_path,
                device=model_config.device
            )
        else:
            from models.model_insightface_dropout import load_insightface_ir50_model
            model = load_insightface_ir50_model(
                model_config.model_path,
                device=model_config.device
            )

    elif model_config.model_type == 'facenet':
        from models.model_facenet import load_facenet_inception_model
        model = load_facenet_inception_model(
            pretrained_type=model_config.pretrained_type,
            device=model_config.device
        )

    elif model_config.model_type == 'emsemble':
        from models.model_facenet import load_facenet_inception_model
        model1 = load_facenet_inception_model(
            pretrained_type=model_config.pretrained_type,
            device=model_config.device
        )
        from models.model_insightface import load_insightface_ir50_model
        model2 = load_insightface_ir50_model(
            model_config.model_path,
            device=model_config.device
        )
        from models.model_insightface_dropout import load_insightface_ir50_model
        model3 = load_insightface_ir50_model(
            model_config.model_path,
            device=model_config.device
        )
        from models.ensemble_model_wrapper import EnsembleModelWrapper
        model = EnsembleModelWrapper([model1, model2, model3])
        return model


    else:
        raise NotImplementedError

    return model


def instantiate_attack(attack_config: BaseAttackConfig,
                       model: BaseAttackModel,
                       ):
    attack = instantiate(attack_config, model=model)

    return attack


def instantiate_dataset_attacker(attacker_config,
                                 attack: BaseAttack = None,
                                 create_attack: Callable = None):

    attacker = instantiate(attacker_config, attack=attack, create_attack=create_attack)

    return attacker