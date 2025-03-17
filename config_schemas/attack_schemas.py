from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from typing import Tuple, Any, Optional, List
from omegaconf import MISSING


@dataclass
class BaseAttackConfig:
    """
    Base class for all attacks schemas
    :param _target_: target class
    :param p: probability of applying the attack
    :param device: device to perform the attack on (cuda or cpu)
    :param decision_threshold: threshold for attack success decision
    :param norm: norm constraint (Linf or L2)
    :param epsilon: maximum allowed perturbation
    """
    _target_: str = MISSING
    p: float = 1.0
    device: str = 'cuda:0'
    decision_threshold: float = 0.5
    norm: str = 'Linf'
    epsilon: float = 0.05
    decision_threshold_margin: float = 0.15


@dataclass
class CWAttackConfig(BaseAttackConfig):
    """
    Configuration for CWAttack
    :param _target_: target class
    :param num_iters: Number of iterations to perform.
    :param c: Weight of the classification loss term.
    :param learning_rate: Learning rate for optimization.
    """
    _target_: str = 'fr_attacks.cw.CWAttack'
    num_iters: Optional[int] = None
    c: float = 1e-1
    learning_rate: float = 1e-2
    early_stopping: bool = False


@dataclass
class DeepFoolAttackConfig(BaseAttackConfig):
    """
    Configuration for DeepFoolAttack
    :param _target_: target class
    :param num_iters: Number of iterations to perform.
    """
    _target_: str = 'fr_attacks.deepfool.DeepFoolCosineAttack'
    num_iters: int = 200


@dataclass
class EvolutionaryAttackConfig(BaseAttackConfig):
    """
    Configuration for EvolutionaryAttack
    :param _target_: target class
    :param num_iters: Number of iterations to perform.
    :param lower_dim: Dimension of the reduced search space.
    """
    _target_: str = 'fr_attacks.evolutionary_attack.EvolutionaryAttack'
    num_iters: int = 200
    lower_dim: int = 112


@dataclass
class JSMAttackConfig(BaseAttackConfig):
    """
    Configuration for JSMAttack
    :param _target_: target class
    :param max_perturbations: Maximum number of pixels to perturb.
    :param theta: Threshold for saliency map computation.
    :param gamma: Threshold for pixel selection.
    """
    _target_: str = 'fr_attacks.jsma.JSMAttack'
    max_perturbations: int = 200
    theta: float = 0.8
    gamma: float = 0.2


@dataclass
class LBFGSAttackConfig(BaseAttackConfig):
    """
    Configuration for LBFGSAttack
    :param _target_: target class
    :param max_iter: Maximum iterations for L-BFGS optimization.
    :param c: Weight of the classification loss term.
    """
    _target_: str = 'fr_attacks.lbfgs.L_BFGSAttack'
    max_iter: int = 100
    c: float = 1e-3


@dataclass
class IFGSMAttackConfig(BaseAttackConfig):
    """
    Configuration for IFGSMAttack
    :param _target_: target class
    :param num_iters: Number of iterations to perform.
    """
    _target_: str = 'fr_attacks.i_fgsm.I_FGSMAttack'
    num_iters: Optional[int] = None
    early_stopping: bool = False


@dataclass
class MIFGSMAttackConfig(BaseAttackConfig):
    """
    Configuration for MIFGSMAttack
    :param _target_: target class
    :param num_iters: Number of iterations to perform.
    :param decay_factor: Decay factor for momentum
    """
    _target_: str = 'fr_attacks.mi_fgsm.MI_FGSMAttack'
    num_iters: Optional[int] = None
    early_stopping: bool = False
    decay_factor: float = 1.0


@dataclass
class DI2FGSMAttackConfig(BaseAttackConfig):
    """
    Configuration for DI2FGSMAttack
    :param _target_: target class
    :param num_iters: Number of iterations to perform.
    :param decay_factor: Decay factor for momentum
    :param transform_prob: Probability of applying DI transform
    """

    _target_: str = 'fr_attacks.di2_fgsm.DI2_FGSMAttack'
    num_iters: Optional[int] = None
    early_stopping: bool = False
    decay_factor: Optional[float] = 0.5
    transform_prob: float = 0.5


@dataclass
class TIFGSMAttackConfig(BaseAttackConfig):
    """
    Configuration for TI_FGSMAttack
    :param _target_: target class
    :param num_iters: Number of iterations to perform.
    :param decay_factor: Decay factor for momentum
    :param transform_prob: Probability of applying DI transform
    :param kernel_size: Size of the Gaussian kernel.
    :param sigma: Standard deviation of the Gaussian kernel.
    """

    _target_: str = 'fr_attacks.ti_fgsm.TI_FGSMAttack'
    num_iters: Optional[int] = None
    early_stopping: bool = False
    decay_factor: Optional[float] = 0.5
    transform_prob: Optional[float] = 0.5
    kernel_size: int = 5
    sigma: float = 1.0


@dataclass
class PIFGSMAttackConfig(BaseAttackConfig):
    """
    Configuration for PI_FGSMAttack
    :param _target_: target class
    :param num_iters: Number of iterations to perform.
    :param decay_factor: Decay factor for momentum
    :param transform_prob: Probability of applying DI transform
    :param kernel_size: Size of the Gaussian kernel.
    :param kernel_size_pi: Size of the patch selection kernel.
    :param sigma: Standard deviation of the Gaussian kernel.
    :param beta: Parameter for the patch selection.
    :param gamma: Parameter for the patch selection.
    """

    _target_: str = 'fr_attacks.pi_fgsm.PI_FGSMAttack'
    num_iters: Optional[int] = None
    early_stopping: bool = False
    decay_factor: float = 0.5
    transform_prob: float = 0.5
    kernel_size: int = 5
    kernel_size_pi: int = 5
    sigma: float = 1.0
    beta: float = 1.0
    gamma: float = 0.1



def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="schemas/attacks", name="base", node=BaseAttackConfig)
    cs.store(group="schemas/attacks", name="cw", node=CWAttackConfig)
    cs.store(group="schemas/attacks", name="deepfool", node=DeepFoolAttackConfig)
    cs.store(group="schemas/attacks", name="evolutionary", node=EvolutionaryAttackConfig)
    cs.store(group="schemas/attacks", name="lbfgs", node=LBFGSAttackConfig)
    cs.store(group="schemas/attacks", name="ifgsm", node=IFGSMAttackConfig)
    cs.store(group="schemas/attacks", name="mifgsm", node=MIFGSMAttackConfig)
    cs.store(group="schemas/attacks", name="di2fgsm", node=DI2FGSMAttackConfig)
    cs.store(group="schemas/attacks", name="ti_fgsm", node=TIFGSMAttackConfig)
    cs.store(group="schemas/attacks", name="pi_fgsm", node=PIFGSMAttackConfig)
    cs.store(group="schemas/attacks", name="jsma", node=JSMAttackConfig)
