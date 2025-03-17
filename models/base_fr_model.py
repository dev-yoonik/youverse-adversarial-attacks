import torch
from torch.nn import functional as F
from typing import Callable
from .base_model import BaseAttackModel


class BaseFRModel(BaseAttackModel):
    def __init__(self, model_,
                 preprocess: Callable=None,
                 device = "cpu"):
        super().__init__(model_, preprocess, device)

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

    def loss_function(self, emb_x1, emb_x2, pair_label):
        """
        Calculate loss based on cosine similarity
        :param emb_x1: embedding 1
        :param emb_x2: embedding 2
        :param pair_label: pairwise label (1 for same identity, 0 for different identity)
        :return: loss value
        """

        # calculate cosine similarity
        cos_sim = self.cosine_similarity(emb_x1, emb_x2)

        # Loss for positive pairs (same identity)
        pos_loss = 1 - cos_sim  # Encourage high similarity

        # Loss for negative pairs (different identity)
        neg_loss = cos_sim  # Enforce margin

        # Combine losses based on labels
        loss = torch.where(pair_label == 1, pos_loss, neg_loss)
        return loss.mean()
