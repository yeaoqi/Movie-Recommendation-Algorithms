
from collections import OrderedDict
import torch
from torch import nn


# AutoEncoder class
class AutoEncoder(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        """
        Initialize the AutoEncoder.

        Args:
            hidden (list): A list of integers specifying the sizes of each layer in the network.
                           The first element is the input size, the last is the bottleneck size.
            dropout (float): Dropout probability to prevent overfitting (default=0.1).
        """
        super(AutoEncoder, self).__init__()

        # Encoder: Build a sequence of layers using OrderedDict.
        d1 = OrderedDict()
        for i in range(len(hidden) - 1):
            # Linear layer from hidden[i] to hidden[i+1].
            d1['enc_linear' + str(i)] = nn.Linear(hidden[i], hidden[i + 1])
            # Optional: Batch normalization (commented out).
            # d1['enc_bn' + str(i)] = nn.BatchNorm1d(hidden[i+1])
            # Dropout layer to randomly drop neurons during training.
            d1['enc_drop' + str(i)] = nn.Dropout(dropout)
            # ReLU activation for non-linearity.
            d1['enc_relu' + str(i)] = nn.ReLU()

        # Sequential container for the encoder layers.
        self.encoder = nn.Sequential(d1)

        # Decoder: Build a sequence of layers using OrderedDict.
        d2 = OrderedDict()
        for i in range(len(hidden) - 1, 0, -1):
            # Linear layer from hidden[i] to hidden[i-1].
            d2['dec_linear' + str(i)] = nn.Linear(hidden[i], hidden[i - 1])
            # Optional: Batch normalization (commented out).
            # d2['dec_bn' + str(i)] = nn.BatchNorm1d(hidden[i - 1])
            # Dropout layer to randomly drop neurons during training.
            d2['dec_drop' + str(i)] = nn.Dropout(dropout)
            # Sigmoid activation for output values in the range [0, 1].
            d2['dec_relu' + str(i)] = nn.Sigmoid()

        # Sequential container for the decoder layers.
        self.decoder = nn.Sequential(d2)

    def forward(self, x):
        """
        Perform a forward pass through the AutoEncoder.
        Args:
            x (torch.Tensor): Input tensor, assumed to be in the range [1, 5].
        Returns:
            torch.Tensor: Reconstructed tensor, also in the range [1, 5].
        """
        # Normalize the input to the range [0, 1].
        x = (x - 1) / 4.0

        # Pass the input through the encoder and then the decoder.
        x = self.decoder(self.encoder(x))

        # Clamp the output to ensure values are in the range [0, 1].
        x = torch.clamp(x, 0, 1.0)

        # Scale the output back to the original range [1, 5].
        x = x * 4.0 + 1

        return x
