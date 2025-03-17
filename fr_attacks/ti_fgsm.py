import torch
import torchvision.transforms.functional as F
import os
import sys
from torchvision.transforms import GaussianBlur
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.di2_fgsm import DI2_FGSMAttack
from models.base_model import BaseAttackModel


class TI_FGSMAttack(DI2_FGSMAttack):
    """
    Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks
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
                 early_stopping: bool = False,
                 kernel_size: int = 5,
                 sigma: float = 1.0,
                 decay_factor: float = 1.0,
                 transform_prob: float = 1.0):
        """
        :param model: BaseAttackModel instance.
        :param p: probability of applying the attack
        :param device: device to perform the attack on (cuda or cpu)
        :param decision_threshold: threshold for attack success decision
        :param norm: norm constraint (Linf or L2)
        :param epsilon: maximum allowed perturbation
        :param num_iters: Number of iterations to perform.
        :param early_stopping: Use to allow early stopping
        :param kernel_size: Size of the Gaussian kernel.
        :param sigma: Standard deviation of the Gaussian kernel.
        :param decay_factor: Decay factor for the momentum term.
        :param transform_prob: Probability of applying transformations.
        """
        super().__init__(p=p, device=device, model=model, num_iters=num_iters, epsilon=epsilon,
                         decay_factor=decay_factor, transform_prob=transform_prob,
                         decision_threshold=decision_threshold, decision_threshold_margin=decision_threshold_margin,
                         norm=norm, early_stopping=early_stopping)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.name = self.name.replace(self.name_2, '')
        self.name_2 = f"TI_FGSM-Iters_{num_iters}-Alpha_{self.alpha}-Kernel_{kernel_size}-Sigma_{sigma}-Decay_{decay_factor}-TP_{transform_prob}"
        self.name += self.name_2

        # Create Gaussian kernel
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.blur = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma)

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

            adversarial_image_t = self.apply_image_transform(adversarial_image)
            adversarial_features_t = self.get_features(adversarial_image_t)

            loss = self.calculate_adversarial_loss(adversarial_features=adversarial_features_t,
                                                   target_features=target_features,
                                                   targeted=self.targeted)

            self.model.zero_grad()
            if adversarial_image_t.grad is not None:
                adversarial_image_t.grad.zero_()

            loss.backward(retain_graph=True)

            if adversarial_image_t.grad is None:
                raise RuntimeError("Gradient calculation failed. Check the computation graph.")

            # Momentum based gradient calculation
            grad = (self.decay_factor * grad +
                    (adversarial_image_t.grad.data / torch.norm(adversarial_image_t.grad.data, p=1)))

            # Smooth gradients with Gaussian blur
            grad_blurred = self.blur(grad)

            # Update perturbation using smoothed gradients
            grad_sign = grad_blurred.sign()

            perturbation = self.alpha * grad_sign

            adversarial_image = adversarial_image + perturbation
            adversarial_image = self.project(adversarial_image, image, self.epsilon)

            if self.early_stopping:
                success = self.check_attack_success(adversarial_image=adversarial_image,
                                                    target_features=target_features,
                                                    targeted=self.targeted)
                if success:
                    break

        adversarial_image = self.prep_data_for_cv2(adversarial_image)
        return adversarial_image
