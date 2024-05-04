import torch
import torch.nn as nn

from batteryml.builders import MODELS
from batteryml.models.nn_model import NNModel


@MODELS.register()
class MyLSTMRULPredictor(NNModel):
    def __init__(self,
                 in_channels: int,
                #  hidden_size1: int = 50,
                #  hidden_size2: int = 100,
                 channels: int,
                 input_height: int ,
                 input_width: int,
                 dropout_prob: float = 0.2,
                 **kwargs):
        NNModel.__init__(self, **kwargs)
        self.lstm1 = nn.LSTM(
            input_size=in_channels * input_width,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            dropout=dropout_prob if dropout_prob > 0 else 0
        )
        self.lstm2 = nn.LSTM(
            input_size=50,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            dropout=dropout_prob if dropout_prob > 0 else 0
        )
        self.fc = nn.Linear(100, 1)

    def forward(self,
                feature: torch.Tensor,
                label: torch.Tensor,
                return_loss: bool = False):
        if feature.ndim == 3:
            feature = feature.unsqueeze(1)
        B, _, H, _ = feature.size()
        x = feature.permute(0, 2, 1, 3).contiguous().view(B, H, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1].contiguous().view(B, -1)
        x = self.fc(x).view(-1)

        if return_loss:
            return torch.mean((x - label.view(-1)) ** 2)

        return x
