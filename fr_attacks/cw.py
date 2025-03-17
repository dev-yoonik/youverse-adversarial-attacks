import torch
import torchvision.transforms.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.base_attack import BaseAttack
from models.base_model import BaseAttackModel


class CWAttack(BaseAttack):
    """
    https://arxiv.org/abs/1608.04644
    """
    def __init__(self,
                 model: BaseAttackModel,
                 p: float = 1.0,
                 device: str = 'cuda:0',
                 decision_threshold: float = 0.5,
                 decision_threshold_margin: float = 0.15,
                 norm: str = 'Linf',
                 epsilon: float = 0.1,
                 c: float = 1e-1,
                 learning_rate: float = 1e-2,
                 num_iters: int = None,
                 early_stopping: bool = False):
        """
        :param model: BaseAttackModel instance.
        :param p: probability of applying the attack
        :param device: device to perform the attack on (cuda or cpu)
        :param decision_threshold: threshold for attack success decision
        :param norm: norm constraint (Linf or L2)
        :param epsilon: maximum allowed perturbation
        :param c: Weight of the classification loss term.
        :param learning_rate: Learning rate for optimization.
        :param num_iters: Number of iterations to perform.
        :param early_stopping: if early stopping is allowed
        """
        super().__init__(p=p, device=device, norm=norm, epsilon=epsilon, decision_threshold=decision_threshold,
                         decision_threshold_margin=decision_threshold_margin)
        self.model = model
        self.target = None
        self.target_processed = None
        self.target_feature = None
        if num_iters is None:
            self.num_iters = int(min((self.epsilon * 255) + 4, (self.epsilon * 255) * 1.25)) # Empirical value.
        else:
            self.num_iters = num_iters
        self.early_stopping = early_stopping
        self.c = c
        self.lr = learning_rate
        self.epsilon = epsilon
        self.norm = norm
        self.name += f"C&W-Iters_{num_iters}-lr_{learning_rate}"

    def get_features(self, x: torch.Tensor):
        """
        Get the features of the model.
        :param x: Input image
        :return: Features
        """
        feat = self.model(x)
        return feat

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
        classification_loss = torch.where(pair_label == 1, pos_loss, neg_loss)

        # Distance loss to minimize the perturbation
        if self.norm == 'L2':
            distance_loss = torch.norm(adversarial_image - target_image, p=2)
        elif self.norm == 'Linf':
            distance_loss = torch.max(torch.abs(adversarial_image - target_image))
        else:
            raise ValueError("Unsupported norm type. Choose 'L2' or 'Linf'.")

        total_loss = self.c * classification_loss + distance_loss
        return total_loss

    @staticmethod
    # Projection function to ensure valid pixel values and bounded perturbation
    def project(adversarial_image, source_image, epsilon):
        """
        Project the perturbed input adversarial_image to ensure the perturbation is within the allowed epsilon range
        and the pixel values are valid (e.g., between 0 and 1).
        :param adversarial_image: adversarial example tensor
        :param source_image: original image
        :param epsilon: Max allowed perturbation
        :return: projected adversarial example
        """
        one, zero = torch.ones_like(source_image), torch.zeros_like(source_image)

        max_inputs = torch.stack([zero, source_image - epsilon, adversarial_image])

        max_val, _ = torch.max(max_inputs, dim=0)

        min_inputs = torch.stack([one, source_image + epsilon, max_val], dim=0)

        adversarial_image, _ = torch.min(min_inputs, dim=0)

        return adversarial_image

    def apply(self, image, target=None, targeted=None, *args, **kwargs):
        """
        Apply the CW attack.
        :param image: Input image tensor.
        :param target: Target embedding or image.
        :param targeted: Whether the attack is targeted.
        :return: Adversarial image tensor.
        """
        target, target_features = self.prep_target(image, target, targeted)

        image = self.prep_data_for_torch(image, add_batch_dim=True).to(self.device)

        adversarial_image = image.clone().detach().requires_grad_(True).to(self.device)
        adversarial_image = adversarial_image + torch.randn_like(adversarial_image) * 1e-3  # Small random perturbation
        adversarial_image = torch.clamp(adversarial_image, min=0, max=1)
        adversarial_image.retain_grad()

        # Initialize optimizer
        optimizer = torch.optim.Adam([adversarial_image], lr=self.lr)

        # Apply attack
        for _ in range(self.num_iters):

            # Compute loss
            optimizer.zero_grad()
            adversarial_features = self.get_features(adversarial_image)

            if self.early_stopping:
                success = self.check_attack_success(adversarial_features=adversarial_features,
                                                    target_features=target_features,
                                                    targeted=self.targeted)
                if success:
                    break

            loss = self.calculate_adversarial_loss(
                adversarial_image=adversarial_image,
                adversarial_features=adversarial_features,
                target=target,
                target_features=target_features,
                targeted=self.targeted)

            # Backprop
            if adversarial_image.grad is not None:
                adversarial_image.grad.zero_()

            loss.backward(retain_graph=True)
            optimizer.step()

            # Here .data is used to avoid computing the gradient for the perturbation
            adversarial_image.data = self.project(adversarial_image.data, image, self.epsilon)

        # Resize
        adversarial_image = self.prep_data_for_cv2(adversarial_image)
        return adversarial_image
