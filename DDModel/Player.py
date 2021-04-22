import torch

from baseline.utils import getOptim
from baseline.baseAgent import baseAgent

from DDModel.PlayerTemp import PlayerTemp


class Player(PlayerTemp):

    def __init__(self, cfg):
        super(Player, self).__init__(cfg)
        self._buildModel()

    def _buildModel(self):
        for netName, data in self._cfg.agent.items():
            if netName == "Embedded":
                self.Embedded = baseAgent(data)
            if netName == "Output":
                self.Output = baseAgent(data)

    def buildOptim(self):
        for key, value in self._cfg.optim.items():
            ebmedded = list(self.Embedded.buildOptim())
            output = list(self.Output.buildOptim())
            for o in output:
                ebmedded.append(o)
            optim = getOptim(value, tuple(ebmedded))
        return optim

    def forward(self, image: torch.tensor, courseActions: torch.tensor) -> torch.tensor:
        # image: b,c,h,w
        # courseActions: S, b, 2
        latent = self.Embedded.forward([image])[0]
        latent = torch.unsqueeze(latent, dim=1)
        self.Output.setCellState((latent, latent))
        sequence = courseActions.shape[0]
        events = []
        for i in range(sequence):
            event = self.Output.forward([courseActions[i, :, :]])[0]
            events.append(event)
        events = torch.cat(events, dim=0)
        return events

    def to(self):
        self.Embedded.to(self._device)
        self.Output.to(self._device)
