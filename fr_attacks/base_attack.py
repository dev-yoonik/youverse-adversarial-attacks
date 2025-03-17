""" Base class for all attacks. """
from abc import abstractmethod
from albumentations.core.transforms_interface import BasicTransform
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Any


class BaseAttack(BasicTransform):
    """
    Base class for all FR attacks.
    Attacks and returns images in cv2 RGB format float [0,1].
    Unbatched processing.
    """
    def __init__(self,
                 p: float,
                 device: str = 'cuda:0',
                 decision_threshold: float = 0.5,
                 norm: str = 'Linf',
                 epsilon: float = 0.1,
                 decision_threshold_margin: float = 0.15):
        """
        :param p: probability of applying the attack
        :param device: device to perform the attack on (cuda or cpu)
        :param decision_threshold: threshold for attack success decision
        :param norm: norm constraint (Linf or L2)
        :param epsilon: maximum allowed perturbation
        :param decision_threshold_margin: margin for decision threshold. It overrides decision_threshold for early stopping.
        """

        super().__init__(p=p)
        if decision_threshold_margin is not None:
            dt_string = "Margin_" + str(decision_threshold_margin)
        else:
            dt_string = "Thresh_" + str(decision_threshold)
        self.name = f"AdvAtk-{dt_string}-Norm_{norm}_{epsilon}-"
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.decision_threshold = decision_threshold
        self.decision_threshold_margin = decision_threshold_margin
        assert 0 <= p <= 1, "Probability should be between 0 and 1."
        self.target, self.target_feature = None, None
        self.targeted = False
        assert norm in ['Linf', 'L2'], "Norm should be either 'Linf' or 'L2'."
        self.norm = norm
        self.epsilon = epsilon
        self.logger = logging.getLogger(__name__)

    @property
    def targets(self):
        return {"image": self.apply, "target": self.identity, "targeted": self.identity}

    @abstractmethod
    def get_features(self, x: torch.Tensor):
        raise NotImplementedError

    def prep_data_for_torch(self, data, add_batch_dim=False):
        # TODO this should be in another class (i.e utils)
        """
        Convert numpy or tensor data to tensor format [C,H,W] and range [0,1].
        Assumes input is in [H,W,C] for numpy and [C,H,W] for tensor.
        """
        if isinstance(data, np.ndarray):
            self.logger.debug(f"Transforming np data to torch tensor")
            if data.dtype == np.uint8:
                data = data.astype(np.float32)
                data /= 255.0
            # If float assume already in range 0-1 and don't divide by 255 else assume uint8 and divide
            data = torch.from_numpy(data).permute(2, 0, 1)


        elif isinstance(data, torch.Tensor):
            self.logger.debug(f"Asserting torch tensor dimensions and range")
            assert data.ndim == 3, "Tensor must have 3 dimensions."  # [C,H,W]
            assert 0 <= data.min() <= 1 and 0 <= data.max() <= 1, "Tensor should be in 0-1 range."
            # To float 32
            data = data.float()
        else:
            raise TypeError("Input data must be numpy.ndarray or torch.Tensor.")

        if add_batch_dim:
            data = data.unsqueeze(0)
        return data

    def prep_data_for_cv2(self, data):
        # TODO this should be in another class (i.e utils)
        """
        Data out should be in cv2 RGB format [0-1]. As such if the input is not in this format it should be converted.
        """
        if isinstance(data, torch.Tensor):
            self.logger.debug(f"Transforming torch tensor to np data")
            if data.ndim == 4:
                data = data.squeeze(0)
            assert data.ndim == 3, "Tensor must have 3 dimensions."
            data = data.cpu().permute(1, 2, 0).squeeze(0).detach().numpy().astype(np.float32)

        assert data.shape[-1] in [1,3,4] and data.ndim == 3, "Wrong shape of data. Should be in [H,W,C]."
        assert data.max() <= 1 and data.min() >= 0 and data.dtype == np.float32, "Wrong range/type of data. Should be in float 0-1 range."
        return (data * 255.0).astype(np.uint8)

    def prep_target(self,
                    adversarial: np.ndarray,
                    target: np.ndarray = None,
                    targeted: bool = None):
        """
        Prepare the target image for the attack. This means extracting the features and storing them. This is stored in
        the class so that attacks with the same target avoid recomputing the features.
        :param adversarial: adversarial image
        :param target: target image
        :param targeted: whether the attack is targeted or not
        :return: normalized image and target features
        """
        if target is None:  # Un-targeted attack -> deviate from self
            target = adversarial
            self.targeted = False  # Negative_pair === target
            self.logger.debug("Targeted attack: False")

        else:  # Targeted attack -> approximate to target
            if targeted is not None:
                self.targeted = targeted
                self.logger.debug(f"Targeted attack: {self.targeted}")
            else:
                self.targeted = True
                self.logger.debug("Targeted attack: True")

        target = self.prep_data_for_torch(target, add_batch_dim=True).to(self.device)
        if self.target_feature is None or not torch.allclose(target, self.target):
            self.target = target
            self.target_feature = self.get_features(self.target)

        return self.target, self.target_feature


    def cosine_similarity(self, x1, x2):
        """
        Calculate cosine similarity between two embeddings
        :param x1: embedding 1
        :param x2: embedding 2
        :return: cosine similarity
        """
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        cos_sim = F.cosine_similarity(x1, x2)
        return cos_sim

    def calculate_adversarial_loss(self,
                       adversarial_image=None,
                       adversarial_features=None,
                       target_image=None,
                       target_features=None,
                       targeted=True):

        if adversarial_features is not None:
            fx_adv = adversarial_features
        elif adversarial_image is not None:
            fx_adv = self.get_features(adversarial_image)
        else:
            raise ValueError("Either adversarial or adversarial_features must be provided.")

        if target_features is not None:
            fx_tgt = target_features
        elif target_image is not None:
            fx_tgt = self.get_features(target_image)
        else:
            raise ValueError("Either target or target_features must be provided.")
        pair_label = 0 if targeted else 1
        pair_label = torch.tensor(pair_label, dtype=torch.long, device=self.device)

        # calculate cosine similarity
        cos_sim = self.cosine_similarity(fx_tgt, fx_adv)

        # Loss for positive pairs (same identity)
        pos_loss = 1 - cos_sim

        # Loss for negative pairs (different identity)
        neg_loss = cos_sim  # Enforce margin

        # Combine losses based on labels
        loss = torch.where(pair_label == 1, pos_loss, neg_loss)

        return loss.mean()

    def calculate_adversarial_loss_2(self,
                       adversarial_image=None,
                       adversarial_features=None,
                       target_image=None,
                       target_features=None,
                       targeted=True):

        if adversarial_features is not None:
            fx_adv = adversarial_features
        elif adversarial_image is not None:
            fx_adv = self.get_features(adversarial_image)
        else:
            raise ValueError("Either adversarial or adversarial_features must be provided.")

        if target_features is not None:
            fx_tgt = target_features
        elif target_image is not None:
            fx_tgt = self.get_features(target_image)
        else:
            raise ValueError("Either target or target_features must be provided.")
        pair_label = 0 if targeted else 1
        pair_label = torch.tensor(pair_label, dtype=torch.long, device=self.device)

        fx_adv, fx_tgt = F.normalize(fx_adv, p=2, dim=1), F.normalize(fx_tgt, p=2, dim=1)
        loss = F.normalize(fx_adv - fx_tgt, p=2, dim=1)

        # Loss for positive pairs (same identity)
        pos_loss = loss  # Encourage high similarity

        # Loss for negative pairs (different identity)
        neg_loss = - loss  # Enforce margin

        # Combine losses based on labels
        loss = torch.where(pair_label == 1, pos_loss, neg_loss)

        return loss.mean()


    def get_params(self):
        """
        Returns parameters for the transform.
        Override if specific parameters are needed for derived attacks.
        """
        return {}

    @abstractmethod
    def apply(self, image, target=None, targeted=None, *args ,**kwargs):
        """
        Attack the given image.
        :param image:
            The image to attack.
        :param target:
            The target image.
        :return:
            The attacked data.
        """
        raise NotImplementedError

    def get_params_dependent_on_data(self, params, data):
        # Extract 'target' and 'targeted' from data and pass it as a parameter
        target = data.get('target', None)
        targeted = data.get('targeted', None)
        return {'target': target, 'targeted': targeted}

    @staticmethod
    def identity(*args, **kwargs):
        """
        Identity function
        :param target:
        :param args:
        :param kwargs:
        :return:
        """
        target = kwargs.get("target", None)
        return target

    def project_perturbation(self, perturbation):
        """
        Project the perturbation to the allowed range.
        :param perturbation: The perturbation to project.
        :return: The projected perturbation.
        """
        if self.norm == 'Linf':
            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
        elif self.norm == 'L2':
            norm = torch.norm(perturbation, p=2, dim=1, keepdim=True)
            factor = torch.clamp(self.epsilon / norm, max=1.0)
            perturbation = perturbation * factor
        else:
            raise NotImplementedError

        return perturbation

    def project_perturbation_from_image(self, image, adversarial_image):
        # Enforce norm constraint
        perturbation = adversarial_image - image
        perturbation = self.project_perturbation(perturbation)
        adversarial_image = image + perturbation
        return adversarial_image

    def check_attack_success(self,
                             adversarial_image: torch.Tensor = None,
                             adversarial_features: torch.Tensor = None,
                             target_image: torch.Tensor = None,
                             target_features: torch.Tensor = None,
                             targeted: bool = False) -> bool:
        """
        Check if the attack was successful.
        :param adversarial_image: adversarial image
        :param adversarial_features: adversarial features
        :param target_image: target image
        :param target_features: target features
        :param targeted: whether this is a targeted attack
        :return: True if the attack was successful, False otherwise
        """

        if adversarial_features is None:
            adversarial_features = self.get_features(adversarial_image)
        if target_features is None:
            target_features = self.get_features(target_image)

        # Norm features
        adversarial_features = F.normalize(adversarial_features, dim=1)
        target_features = F.normalize(target_features, dim=1)

        # Calculate similarity
        similarity = F.cosine_similarity(adversarial_features, target_features, dim=1)

        if targeted:
            if self.decision_threshold_margin is not None:
                success_bool = (similarity > 1 - self.decision_threshold_margin).item()
            else:
                success_bool = (similarity > self.decision_threshold).item()
        else:
            if self.decision_threshold_margin is not None:
                success_bool = (similarity < 0 + self.decision_threshold_margin).item()
            else:
                success_bool = (similarity < self.decision_threshold).item()

        if success_bool:
            self.logger.debug(f"Attack was successful. Similarity: {similarity.item()}")

        return success_bool


    def check_final_similarity(self,
                             adversarial_image: torch.Tensor = None,
                             adversarial_features: torch.Tensor = None,
                             target_image: torch.Tensor = None,
                             target_features: torch.Tensor = None):
        """
        Check if the attack was successful.
        :param adversarial_image: adversarial image
        :param adversarial_features: adversarial features
        :param target_image: target image
        :param target_features: target features
        :return: similarity value
        """

        if adversarial_features is None:
            adversarial_features = self.get_features(adversarial_image)
        if target_features is None:
            target_features = self.get_features(target_image)

        # Norm features
        adversarial_features = F.normalize(adversarial_features, dim=1)
        target_features = F.normalize(target_features, dim=1)

        # Calculate similarity
        similarity = F.cosine_similarity(adversarial_features, target_features, dim=1)
        return similarity
