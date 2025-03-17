import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.base_attack import BaseAttack
from models.base_model import BaseAttackModel


class L_BFGSAttack(BaseAttack):
    """
    L-BFGS-based adversarial attack for face recognition models.
    """

    def __init__(self,
                 model: BaseAttackModel,
                 p: float = 1.0,
                 device: str = 'cuda:0',
                 decision_threshold: float = 0.5,
                 decision_threshold_margin: float = 0.15,
                 norm: str = 'Linf',
                 epsilon: float = 0.1,
                 max_iter: int = 100,
                 c: float = 1e-3):
        """
        :param model: BaseAttackModel instance.
        :param p: probability of applying the attack
        :param device: device to perform the attack on (cuda or cpu)
        :param decision_threshold: threshold for attack success decision
        :param norm: norm constraint (Linf or L2)
        :param epsilon: maximum allowed perturbation
        :param max_iter: Maximum iterations for L-BFGS optimization.
        :param c: Regularization parameter for image similarity.
        """
        super().__init__(p=p, device=device, norm=norm, epsilon=epsilon,
                         decision_threshold=decision_threshold, decision_threshold_margin=decision_threshold_margin)
        self.model = model
        self.lr = 0.05
        self.num_iters = max_iter
        self.c = c
        self.name += f"L_BFGS-Iters_{max_iter}-C_{c}"

    def get_features(self, x: torch.Tensor):
        feat = self.model(x)
        return feat

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
        Apply the attack.
        :param image: Input image tensor.
        :param target: Target embedding or image.
        :param targeted: Whether the attack is targeted.
        :return: Adversarial image tensor.
        """
        target, target_features = self.prep_target(image, target, targeted)
        # Prep data
        image = self.prep_data_for_torch(image, add_batch_dim=True).to(self.device)

        # Clone inputs so that we don't modify the originals
        adversarial_image = image.clone().detach().requires_grad_(True).to(self.device)

        # Add a small initial random perturbation
        initial_perturbation = torch.empty_like(adversarial_image).uniform_(-1e-3, 1e-3)
        adversarial_image = torch.clamp(adversarial_image + initial_perturbation, min=0, max=1).detach()
        adversarial_image.requires_grad_(True)

        optimizer = torch.optim.LBFGS([adversarial_image], lr=self.lr, max_iter=self.num_iters, tolerance_grad=1e-5)

        def closure():
            """Optimizer closure to compute loss and gradients."""
            if torch.is_grad_enabled():
                optimizer.zero_grad()  # Clear previous gradients

            adversarial_features = self.get_features(adversarial_image)

            # Compute the feature loss
            loss = self.calculate_adversarial_loss(
                adversarial_features=adversarial_features,
                target_features=target_features,
                targeted=self.targeted,
            )

            if loss.requires_grad:
                loss.backward(retain_graph=True)  # Compute gradients

                # Ensure gradients are contiguous
                for p in optimizer.param_groups[0]["params"]:
                    if p.grad is not None and not p.grad.is_contiguous():
                        p.grad = p.grad.contiguous()

            return loss

        # Run the L-BFGS optimizer
        optimizer.step(closure)

        # Process the adversarial image for output
        adversarial_image = adversarial_image.detach()  # Detach the tensor from the computation graph
        adversarial_image = self.project(adversarial_image, image, self.epsilon)

        adversarial_image = self.prep_data_for_cv2(adversarial_image)  # Convert to CV2 format
        return adversarial_image
