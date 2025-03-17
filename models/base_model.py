import torch
from typing import Callable
from abc import abstractmethod


class BaseAttackModel(torch.nn.Module):

    def __init__(self, model_, preprocess: Callable=None, device = "cpu"):
        super(BaseAttackModel, self).__init__()
        self.device = torch.device(device)

        # prep model
        self.model = model_
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.to(device)

        # get data preprocess
        self.preprocess = preprocess

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        pass

    def forward(self, x: torch.Tensor):
        if self.preprocess is not None:
            x = self.preprocess(x)
        x = self.model(x)
        return x