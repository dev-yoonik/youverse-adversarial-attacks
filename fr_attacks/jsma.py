import torch
import numpy as np
import torchvision.transforms.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.base_attack import BaseAttack
from models.base_model import BaseAttackModel


class JSMAttack(BaseAttack):
    """
    Jacobian-based Saliency Map Attack (JSMA) for Face Recognition Models
    Adapted from the original JSMA concept to work with embedding similarity
    """

    def __init__(self,
                 model: BaseAttackModel,
                 p: float = 1.0,
                 device: str = 'cuda:0',
                 decision_threshold: float = 0.5,
                 decision_threshold_margin: float = 0.15,
                 norm: str = 'Linf',
                 epsilon: float = 0.1,
                 max_perturbations: int = 200,
                 theta: float = 0.8,
                 gamma: float = 0.2):
        """
        Initialize the JSMA Attack.

        :param model: BaseAttackModel instance.
        :param max_perturbations: Maximum number of pixels to perturb.
        :param theta: Threshold for saliency map computation.
        :param gamma: Step size for pixel perturbation.
        :param device: Device to perform the attack on.
        :param p: Lp norm parameter.
        """
        super().__init__(p=p, device=device, decision_threshold=decision_threshold, decision_threshold_margin=decision_threshold_margin, norm=norm, epsilon=epsilon)
        self.model = model

        self.max_perturbations = max_perturbations
        self.theta = theta
        self.gamma = gamma
        self.name += f"JSMA-Perturbs_{max_perturbations}-Theta_{theta}"

    def get_features(self, x: torch.Tensor):
        feat = self.model(x)
        return feat

    def calculate_embedding_similarity(self, x1_features, x2_features, negative_pair=True):
        """
        Calculate similarity between two face embeddings.

        :param x1_features: Features of the first image
        :param x2_features: Features of the second image (target)
        :param negative_pair: Whether this is a negative pair (want to increase dissimilarity)
        :return: Similarity score
        """
        pair_label = 0 if negative_pair else 1
        pair_label = torch.tensor(pair_label, dtype=torch.long, device=self.device)

        # Calculate the loss (similarity)
        loss = self.model.loss_function(x1_features, x2_features, pair_label=pair_label)

        return loss

    def compute_saliency_map(self, image, target_features, targeted):
        """
        Compute the saliency map based on embedding similarity changes.

        :param image: Input image tensor
        :param target_features: Target embedding features
        :param targeted: Whether this is a negative pair attack
        :return: Saliency map tensor
        """
        # Ensure gradient computation
        x_perturb = image.clone().detach().requires_grad_(True)

        # Compute gradients
        self.model.zero_grad()
        x_perturb.retain_grad()

        # Compute similarity loss
        loss = self.calculate_embedding_similarity(
            self.get_features(x_perturb),
            target_features,
            negative_pair=targeted
        )

        # Backpropagate
        loss.backward(retain_graph=True)

        # Get gradient of each pixel
        saliency_map = torch.abs(x_perturb.grad)

        return saliency_map

    def apply(self, image, target=None, targeted=None, *args, **kwargs):
        """
        Apply the attack.
        :param image: Input image tensor.
        :param target: Target embedding or image.
        :param targeted: Whether the attack is targeted.
        :return: Adversarial image tensor.
        """
        target, target_features = self.prep_target(image, target, targeted)

        # Prepare input
        image = self.prep_data_for_torch(image, add_batch_dim=True).to(self.device)

        # Create a copy of the image to modify
        adversarial_image = image.clone()

        # Perform iterative perturbation
        for _ in range(self.max_perturbations):
            # Compute saliency map
            saliency_map = self.compute_saliency_map(adversarial_image, target_features, self.targeted)

            # Per-channel perturbation strategy
            saliency_map_channels = saliency_map.squeeze(0)  # Shape: [3, H, W]

            # For each channel, find the most salient pixel
            channel_pixel_indices = []
            for channel in range(3):
                channel_saliency = saliency_map_channels[channel]

                pixel_to_perturb = torch.argmax(channel_saliency)

                # Convert to 2D indices
                y = pixel_to_perturb // channel_saliency.shape[-1]
                x = pixel_to_perturb % channel_saliency.shape[-1]

                channel_pixel_indices.append((channel, y, x))

            # Perturb pixels based on per-channel saliency
            for channel, y, x in channel_pixel_indices:
                current_value = adversarial_image[0, channel, y, x]
                new_value = torch.clamp(current_value + self.gamma, 0, 1)
                adversarial_image[0, channel, y, x] = new_value

            success = self.check_attack_success(adversarial_image=adversarial_image,
                                                target_features=target_features,
                                                targeted=self.targeted)
            if success:
                break

        adversarial_image = self.prep_data_for_cv2(adversarial_image)

        return adversarial_image