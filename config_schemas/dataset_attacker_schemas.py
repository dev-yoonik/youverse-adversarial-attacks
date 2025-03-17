from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from typing import Any, Optional, List, Callable
from omegaconf import MISSING


@dataclass
class BaseDatasetAttackerConfig:
    """
    Base class for all dataset attacker schemas
    :param _target_: target class
    """
    _target_: str = MISSING

@dataclass
class DatasetAttackerConfig(BaseDatasetAttackerConfig):
    """
     Dataset attacker
    :param _target_: target class
    :param dataset_root: root path of the dataset
    :param attack: BaseAttack instance
    :param random_percentage: percentage of the dataset to attack
    """

    _target_: str = "datasets.attack_dataset.BaseDatasetAttacker"
    dataset_root: str = MISSING
    attack: Any = MISSING
    random_percentage: float = None


@dataclass
class MultiThreadedDatasetAttackerConfig(BaseDatasetAttackerConfig):
    """
    Multi-threaded dataset attacker configuration
    :param _target_: target class
    :param num_threads: number of threads to use
    :param dataset_root: root path of the dataset
    :param attack_name: name of the attack
    :param random_percentage: percentage of the dataset to attack
    :param bona_fide_paths: list of bona fide image paths
    :param target_paths: list of target image paths
    :param targeted_list: list of booleans indicating whether the attack is targeted
    :param create_attack: function to create the attack
    :param create_attack_directory: whether to create the attack directory
    """
    _target_: str = "datasets.attack_dataset.MultiThreadedDatasetAttacker"
    dataset_root: str = MISSING
    attack_name: Optional[str] = ""
    num_threads: int = 4
    random_percentage: Optional[float] = None
    bona_fide_paths: Optional[list] = None
    target_paths: Optional[list] = None
    targeted_list: Optional[List[bool]] = None
    create_attack: Any = MISSING
    create_attack_directory: bool = True
    attack_dataset_root: Optional[str] = None
    logging_path: Optional[str] = None


@dataclass
class LFWBenchmarkAttackerMultithreadedConfig(BaseDatasetAttackerConfig):
    """
    Configuration for LFWBenchmarkAttackerMultithreaded
    """
    _target_: str = "datasets.attack_lfw_benchmark.LFWBenchmarkAttackerMultithreaded"
    dataset_root: str = MISSING
    attack_name: Optional[str] = ""
    pairs_txt_path: str = MISSING
    num_threads: int = 4
    random_percentage: Optional[float] = None
    create_attack: Any = MISSING
    attack_dataset_root: Optional[str] = None
    logging_path: Optional[str] = None


@dataclass
class CelebAAttackerMultithreadedConfig(BaseDatasetAttackerConfig):
    """
    Configuration for CelebAAttackerMultithreaded
    """
    _target_: str = "datasets.attack_celeba.CelebAAttackerMultithreaded"
    dataset_root: str = MISSING
    attack_label_list: str = MISSING
    num_threads: int = 4
    create_attack: Any = MISSING
    create_attack_directory: bool = True
    attack_dataset_root: Optional[str] = None
    logging_path: Optional[str] = None


def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="schemas/dataset_attackers", name="base", node=BaseDatasetAttackerConfig)
    cs.store(group="schemas/dataset_attackers", name="multithreaded", node=MultiThreadedDatasetAttackerConfig)
    cs.store(group="schemas/dataset_attackers", name="lfw_benchmark_multithreaded", node=LFWBenchmarkAttackerMultithreadedConfig)
    cs.store(group="schemas/dataset_attackers", name="celeba_multithreaded", node=CelebAAttackerMultithreadedConfig)

