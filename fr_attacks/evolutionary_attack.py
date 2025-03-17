import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr_attacks.base_attack import BaseAttack
from models.base_model import BaseAttackModel
import torch
import numpy as np
from scipy.ndimage import zoom

class EvolutionaryAttack(BaseAttack):
    def __init__(self,
                 model: BaseAttackModel,
                 p: float = 1.0,
                 device: str = 'cuda:0',
                 decision_threshold: float = 0.5,
                 decision_threshold_margin: float = 0.15,
                 norm: str = 'Linf',
                 epsilon: float = 0.1,
                 num_iters=200,
                 lower_dim=224):
        """
        :param model: BaseAttackModel instance.
        :param p: probability of applying the attack
        :param device: device to perform the attack on (cuda or cpu)
        :param decision_threshold: threshold for attack success decision
        :param norm: norm constraint (Linf or L2)
        :param epsilon: maximum allowed perturbation
        :param num_iters: Number of iterations to perform.
        :param lower_dim: Dimension of the reduced search space.
        """
        super().__init__(p=p, device=device, norm=norm, epsilon=epsilon, decision_threshold=decision_threshold,
                         decision_threshold_margin=decision_threshold_margin)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.num_iters = num_iters
        self.noise_scale = epsilon
        self.lower_dim = lower_dim
        self.name += f"Evolutionary-Iters{num_iters}-Noise_scale_{epsilon}-Lower_dim_{lower_dim}"

    def get_features(self, x: torch.Tensor):
        feat = self.model(x)
        return feat

    def apply(self, image, target=None, targeted=None, *args, **kwargs):
        """
        Apply the attack.
        :param image: Input image tensor.
        :param target: Target embedding or image.
        :param targeted: Whether the attack is targeted.
        :return: Adversarial image tensor.
        """
        target, target_features = self.prep_target(image, target, targeted)

        image_original = self.prep_data_for_torch(image, add_batch_dim=True).to(self.device)
        input_shape = image_original.shape

        adversarial_image = image_original.clone().detach().to(self.device)
        parent = adversarial_image.clone().detach().to(self.device)
        parent_fitness = None

        with torch.no_grad():
            for i in range(self.num_iters):
                # Generate offspring by adding random perturbation
                perturbation = self.sample_noise(input_shape)
                offspring = self.project(image_original + perturbation, image_original, self.epsilon)

                # Evaluate the fitness of parent and offspring
                if parent_fitness is None:
                    parent_fitness = self.evaluate_fitness(parent, target_features, self.targeted)
                offspring_fitness = self.evaluate_fitness(offspring, target_features, self.targeted)

                # If offspring has better fitness, replace the parent
                if offspring_fitness > parent_fitness:
                    parent = offspring
                    parent_fitness = offspring_fitness

                    success = self.check_attack_success(parent,
                                                        target_features=target_features,
                                                        targeted=self.targeted)
                    if success:
                        break

            return self.prep_data_for_cv2(parent)

    def sample_noise(self, input_shape):
        """
        Sample noise from a reduced lower-dimensional space and upscale it.
        """
        b, c, h, w = input_shape
        # Sample noise in the reduced dimensional space
        noise_low_dim = np.random.randn(c, self.lower_dim, self.lower_dim)
        # Upscale noise to the original dimensions
        noise = zoom(noise_low_dim, (1, h / self.lower_dim, w / self.lower_dim), order=1)
        return torch.from_numpy(self.noise_scale * noise).unsqueeze(0).to(self.device).float()

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

    def evaluate_fitness(self, adversarial_example, target_feature, targeted):
        """
        Evaluate the fitness by calculating the model's output for the adversarial example.
        The goal is to miss-classify or target a specific label.
        """
        loss = self.calculate_adversarial_loss(adversarial_image=adversarial_example, target_features=target_feature, targeted=targeted)
        return loss

