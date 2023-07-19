import torch
import torch.nn.functional as F
from torch.optim import Adam
from trainer.trainer import Trainer


class MapTrainer(Trainer):
    def __init__(self, **kwargs):
        super(MapTrainer, self).__init__(main_metric='mae',
                                          optim=Adam,
                                          main_metric_goal='min',
                                          **kwargs)

    def forward_pass(self, batch):
        rgraphs, pgraphs, targets, mapping = tuple(batch)
        y_pred = self.model(rgraphs, pgraphs)

        if mapping is None:
            raise RuntimeError
        natmax = max(map(len, mapping))

        z = torch.zeros((len(mapping), natmax, natmax), device=self.device)
        for i, map_i in enumerate(mapping):
            for r, p in enumerate(map_i):
                z[i, r, p] = 1.0

        loss = self.loss_func(y_pred, z)

        return loss, y_pred, z
