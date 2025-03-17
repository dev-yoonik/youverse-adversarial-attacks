import torch
import torchvision.transforms.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.base_attack import BaseAttack
from models.base_model import BaseAttackModel


class I_FGSMAttack(BaseAttack):
    """
    Iterative Fast Gradient Sign Method (I-FGSM)
    """

    def __init__(self,
                 model: BaseAttackModel,
                 p: float = 1.0,
                 device: str = 'cuda:0',
                 decision_threshold: float = 0.5,
                 decision_threshold_margin: float = 0.15,
                 norm: str = 'Linf',
                 epsilon: float = 0.1,
                 num_iters: int = None,
                 early_stopping: bool = False):
        """
        :param model: BaseAttackModel instance.
        :param p: probability of applying the attack
        :param device: device to perform the attack on (cuda or cpu)
        :param decision_threshold: threshold for attack success decision
        :param norm: norm constraint (Linf or L2)
        :param epsilon: maximum allowed perturbation
        :param num_iters: Number of iterations to perform.
        :param early_stopping: Use to allow early stopping
        """
        super().__init__(p=p, device=device, norm=norm, epsilon=epsilon, decision_threshold=decision_threshold, decision_threshold_margin=decision_threshold_margin)
        self.model = model
        if num_iters is None:
            self.num_iters = int(min((self.epsilon * 255) + 4, (self.epsilon * 255) * 1.25)) # Empirical value.
        else:
            self.num_iters = num_iters
        self.logger.debug(f"Attacking {self.num_iters} iters.")
        self.alpha = epsilon / self.num_iters
        self.name_2 = f"I_FGSM-Iters_{num_iters}-Alpha_{self.alpha}"
        self.name += self.name_2
        self.early_stopping = early_stopping

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
        adversarial_image.retain_grad()

        sim = self.check_final_similarity(adversarial_image=adversarial_image, target_features=target_features)
        self.logger.debug(f"Initial {sim}")

        for i in range(self.num_iters):

            adversarial_features = self.get_features(adversarial_image)
            if self.early_stopping:
                success = self.check_attack_success(adversarial_features=adversarial_features,
                                                    target_features=target_features,
                                                    targeted=self.targeted)
                if success:
                    break

            # Compute the feature loss
            loss = self.calculate_adversarial_loss(adversarial_features=adversarial_features,
                                       target_features=target_features,
                                       targeted=self.targeted)


            # Zero the gradients from the previous step to prevent accumulation
            self.model.zero_grad()
            if adversarial_image.grad is not None:
                adversarial_image.grad.zero_()

            # Backpropagate the loss to compute gradients
            loss.backward(retain_graph=True)

            # Check if gradients are being computed
            if adversarial_image.grad is None:
                raise RuntimeError("Gradient calculation failed. Check the computation graph.")

            # Update the perturbation using the sign of the gradients
            grad_sign = adversarial_image.grad.data.sign()
            perturbation = self.alpha * grad_sign

            # Project the perturbation using the projection function
            adversarial_image = adversarial_image + perturbation
            adversarial_image = self.project(adversarial_image, image, self.epsilon)

            # Re-enable gradient tracking for the next iteration
            adversarial_image.requires_grad_(True)
            adversarial_image.retain_grad()

            # Logg last results
            self.logger.debug(f"Loss {loss.item()}")
            sim = self.check_final_similarity(adversarial_image=adversarial_image, target_features=target_features)
            self.logger.debug(f"Sim {sim}")

        adversarial_image = self.prep_data_for_cv2(adversarial_image)

        return adversarial_image