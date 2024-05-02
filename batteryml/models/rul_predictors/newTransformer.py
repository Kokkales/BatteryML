import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from batteryml.builders import MODELS
from batteryml.models.nn_model import NNModel


@MODELS.register()
class TransformerRULPredictor(NNModel):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 **kwargs):
        super(TransformerRULPredictor, self).__init__(**kwargs)
        self.transformer = TransformerModel(
            in_channels * input_width, d_model, num_heads, num_layers, dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self,
                feature: torch.Tensor,
                label: torch.Tensor,
                return_loss: bool = False):
        if feature.ndim == 3:
            feature = feature.unsqueeze(1)
        B, _, H, _ = feature.size()
        x = feature.permute(0, 2, 1, 3).contiguous().view(B, H, -1)
        x = self.transformer(x)
        x = x[:, -1].contiguous().view(B, -1)
        x = self.fc(x).view(-1)

        if return_loss:
            return torch.mean((x - label.view(-1)) ** 2)

        return x

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x
