import torch
import torchvision.transforms.functional as F
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.mi_fgsm import MI_FGSMAttack
from models.base_model import BaseAttackModel


class DI2_FGSMAttack(MI_FGSMAttack):
    """
    Improving Transferability of Adversarial Examples with Input Diversity
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
        :param decay_factor: Decay factor for momentum
        :param transform_prob: Probability of applying DI transform
        """
        super().__init__(p=p, model=model, device=device, decision_threshold=decision_threshold,
                         norm=norm, epsilon=epsilon, num_iters=num_iters, decay_factor=decay_factor,
                         decision_threshold_margin=decision_threshold_margin, early_stopping=early_stopping)

        self.name = self.name.replace(self.name_2, '')
        self.name_2 = f"DI2_FGSM-Iters_{num_iters}-Alpha_{self.alpha}-Decay_{decay_factor}-Trans_prob_{transform_prob}"
        self.name += self.name_2

        self.transforms = A.Compose([A.OneOf([
                                   A.ImageCompression(quality_lower=80, quality_upper=100, p=1.0),
                                   A.RandomResizedCrop(height=112, width=112, scale=(0.7, 1.0), p=1.0),
                                   A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                                   A.HorizontalFlip(p=1.0),
                                   A.Rotate(limit=(-10, 10), interpolation=1, border_mode=0, p=1.0)], p=transform_prob),
                                   ToTensorV2(p=1.0)])

    def apply_image_transform(self, image):

        # Apply DI transform to adversarial example
        adversarial_image_t = image.clone()
        adversarial_image_t = adversarial_image_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        adversarial_image_t = self.transforms(image=adversarial_image_t)['image']
        adversarial_image_t = adversarial_image_t.unsqueeze(0).to(self.device).detach()

        # Replace adversarial example with transformed image
        adversarial_image_t = adversarial_image_t.clone().detach().requires_grad_(True)
        adversarial_image_t.retain_grad()

        return adversarial_image_t

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

            # Zero the gradients from the previous step to prevent accumulation
            self.model.zero_grad()
            if adversarial_image_t.grad is not None:
                adversarial_image_t.grad.zero_()

            # Backpropagate the loss to compute gradients
            loss.backward(retain_graph=True)

            # Momentum based gradient calculation
            grad = (self.decay_factor * grad +
                    (adversarial_image_t.grad.data / torch.norm(adversarial_image_t.grad.data, p=1)))

            # Compute the perturbation
            grad_sign = grad.sign()
            perturbation = self.alpha * grad_sign
            adversarial_image = adversarial_image + perturbation

            # Re-enable gradient tracking for the next iteration
            adversarial_image = self.project(adversarial_image, image, self.epsilon)

            if self.early_stopping:
                success = self.check_attack_success(adversarial_image=adversarial_image,
                                                    target_features=target_features,
                                                    targeted=self.targeted)
                if success:
                    break

        adversarial_image = self.prep_data_for_cv2(adversarial_image)
        return adversarial_image