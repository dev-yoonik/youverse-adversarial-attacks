import os
import argparse
import hydra
from omegaconf import OmegaConf
from functools import partial
from config_schemas.instantiate import instantiate_attack, instantiate_model, instantiate_dataset_attacker
from config_schemas.register import register_config
import multiprocessing
import logging

main_logger = logging.getLogger(__name__)

register_config()

# Custom resolver to get file name from path
OmegaConf.register_new_resolver("basename", lambda path: os.path.basename(path))
OmegaConf.register_new_resolver("attack_name", lambda path: path.split(".")[-1])


def create_attack(config):
    model = instantiate_model(config.model)

    attack = instantiate_attack(config.attack, model=model)

    return attack

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="./configs", help='Path to json config.')
parser.add_argument('--config_name', type=str, default="config", help='config name.')
args = parser.parse_args()


@hydra.main(
    version_base=None,
    config_path=args.config_path,
    config_name=args.config_name)
def main(config):

    # Step 2: Instantiate class and determine correct folder name
    attack = create_attack(config)
    attack_name = attack.name
    del attack
    config.dataset_attacker.attack_dataset_root = os.path.join(config.save_dir, "data")
    config.dataset_attacker.attack_name = attack_name + config.dataset_attacker.attack_name
    config.dataset_attacker.logging_path = os.path.join(config.save_dir, "attack.log")

    dataset_attacker = instantiate_dataset_attacker(config.dataset_attacker,
                                                    create_attack=partial(create_attack, config))
    dataset_attacker.attack_dataset()

    main_logger.debug("--------------------Attack complete--------------------------")

if __name__ == "__main__":
    import os
    import warnings
    import sys
    if sys.platform != "win32" and multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    warnings.filterwarnings("ignore")
    main()
