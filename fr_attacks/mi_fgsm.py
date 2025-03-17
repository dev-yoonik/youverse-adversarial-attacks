import torch
import torchvision.transforms.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.i_fgsm import I_FGSMAttack
from models.base_model import BaseAttackModel


class MI_FGSMAttack(I_FGSMAttack):
    """ Boosting Adversarial Attacks with Momentum """
    def __init__(self,
                 model: BaseAttackModel,
                 p: float = 1.0,
                 device: str = 'cuda:0',
                 decision_threshold: float = 0.5,
                 decision_threshold_margin: float = 0.15,
                 norm: str = 'Linf',
                 epsilon: float = 0.1,
                 num_iters: int = None,
                 early_stopping: bool = False,
                 decay_factor: float = 1.0):
        """
        :param model: BaseAttackModel instance.
        :param p: probability of applying the attack
        :param device: device to perform the attack on (cuda or cpu)
        :param decision_threshold: threshold for attack success decision
        :param norm: norm constraint (Linf or L2)
        :param epsilon: maximum allowed perturbation
        :param num_iters: Number of iterations to perform.
        :param early_stopping: Use to allow early stopping
        :param decay_factor: Decay factor for momentum
        """
        super().__init__(p=p, model=model, device=device, decision_threshold=decision_threshold,
                         decision_threshold_margin=decision_threshold_margin,
                         norm=norm, epsilon=epsilon, num_iters=num_iters, early_stopping=early_stopping)

        self.decay_factor = decay_factor

        self.name = self.name.replace(self.name_2, '')
        self.name_2 = f"MI_FGSM-Iters_{num_iters}-Alpha_{self.alpha}-Decay_{decay_factor}"
        self.name += self.name_2

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

        # Initialize grad
        grad = 0

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

            # Momentum based gradient calculation
            grad = self.decay_factor * grad + (adversarial_image.grad.data / torch.norm(adversarial_image.grad.data, p=1))

            # Compute the perturbation
            grad_sign = grad.sign()
            perturbation = self.alpha * grad_sign

            # Update the adversarial image
            adversarial_image = adversarial_image + perturbation
            adversarial_image = self.project(adversarial_image, image, self.epsilon)

            # Re-enable gradient tracking for the next iteration
            adversarial_image.requires_grad_(True)
            adversarial_image.retain_grad()

        adversarial_image = self.prep_data_for_cv2(adversarial_image)
        return adversarial_image
