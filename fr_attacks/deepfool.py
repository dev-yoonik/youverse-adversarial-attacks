import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.base_attack import BaseAttack
from models.base_model import BaseAttackModel


class DeepFoolCosineAttack(BaseAttack):
    """
    DeepFool-based Attack for Face Recognition with Cosine Similarity Threshold
    """
    def __init__(self,
                 model: BaseAttackModel,
                 p: float = 1.0,
                 device: str = 'cuda:0',
                 decision_threshold: float = 0.5,
                 decision_threshold_margin: float = 0.15,
                 norm: str = 'Linf',
                 epsilon: float = 0.1,
                 num_iters: int = 200):
        """
        :param model: BaseAttackModel instance.
        :param p: probability of applying the attack
        :param device: device to perform the attack on (cuda or cpu)
        :param decision_threshold: threshold for attack success decision
        :param norm: norm constraint (Linf or L2)
        :param epsilon: maximum allowed perturbation
        :param num_iters: Maximum number of iterations to perform.
        """
        super().__init__(p=p, device=device, norm=norm, epsilon=epsilon, decision_threshold=decision_threshold,
                         decision_threshold_margin=decision_threshold_margin)
        self.model = model
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.name += f"DeepFool_Cosine-Iters_{num_iters}"

    def get_features(self, x: torch.Tensor):
        """ Extract normalized feature embeddings. """
        feat = self.model(x)
        return F.normalize(feat, p=2, dim=1)


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
        Apply the DeepFool-based cosine similarity attack.
        :param image: Input image tensor.
        :param target: Target embedding or image.
        :param targeted: Whether the attack is targeted.
        :return: Adversarial image tensor.
        """
        target, target_features = self.prep_target(image, target, targeted)

        # Prep data
        image = self.prep_data_for_torch(image, add_batch_dim=True).to(self.device)

        # Clone inputs
        adversarial_image = image.clone().detach().requires_grad_(True).to(self.device)
        adversarial_image.retain_grad()

        for i in range(self.num_iters):
            # Compute embeddings
            adversarial_features = self.get_features(adversarial_image)

            # Cosine similarity
            cosine_sim = F.cosine_similarity(adversarial_features, target_features, dim=1)

            # Stopping criterion
            if cosine_sim.item() < self.decision_threshold:
                break

            # Compute gradient of cosine similarity w.r.t. input
            if self.targeted:
                loss = cosine_sim
            else:
                loss = 1 - cosine_sim

            self.model.zero_grad()
            adversarial_image.grad = None
            loss.backward(retain_graph=True)

            # Compute minimal perturbation direction (DeepFool step)
            grad = adversarial_image.grad.data
            grad_norm = torch.norm(grad, p=2) + 1e-8  # L2 norm
            minimal_step = grad / grad_norm

            perturbation = minimal_step * self.epsilon
            adversarial_image = adversarial_image + perturbation

            # Project perturbation to ensure it is within epsilon-ball
            adversarial_image = self.project(adversarial_image, image, self.epsilon)
            adversarial_image.requires_grad_(True)
            adversarial_image.retain_grad()

        adversarial_image = self.prep_data_for_cv2(adversarial_image)
        return adversarial_image
