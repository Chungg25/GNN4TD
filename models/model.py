from typing import Tuple
from .CCRNN.evonn2 import EvoNN2
from .GMSDR.GMSDR import GMSDRNet
import torch
from torch import nn, Tensor



def create_model(model_name, loss, conf, device,  support=None):
    support = torch.tensor(support, dtype=torch.float32)  
    if model_name == 'CCRNN':
        model = EvoNN2(**conf, support=support, device=device)
        return model, MetricNNTrainer(model, loss, model_name)
    if model_name == 'GMSDR':
        model = GMSDRNet(**conf, support=support, device=device)
        return model, MetricNNTrainer(model, loss, model_name)




class Trainer:
    def __init__(self, model: nn.Module, loss):
        self.model = model
        self.loss = loss

    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        raise ValueError('Not implemented.')


class MetricNNTrainer(Trainer):
    def __init__(self, model, loss, model_name=None):
        super(MetricNNTrainer, self).__init__(model, loss)
        self.model_name = model_name
        self.train_batch_seen: int = 0

    def train(self, inputs: Tensor, targets: Tensor, phase: str):
        outputs, graph = self.model(inputs)
        loss = self.loss(outputs, targets, graph)
        if self.model_name == 'CCRNN':
            if phase == 'train':
                self.train_batch_seen += 1
            i_targets = targets if phase == 'train' else None
            outputs, graph = self.model(inputs, i_targets, self.train_batch_seen)
            loss = self.loss(outputs, targets, graph)
 
        return outputs, loss

