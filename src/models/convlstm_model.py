"""
src/models/convlstm_model.py
============================
PyTorch ConvLSTM for spatiotemporal wildfire ignition prediction.

Architecture
------------
Input : (batch, seq_len, C, H, W)  — sequence of spatial feature maps
Output: (batch, 1, H, W)           — per-pixel ignition probability map

ConvLSTMCell
  Replaces the matrix multiplications in a standard LSTM cell with 2-D
  convolutions, allowing the model to learn local spatial dependencies
  alongside temporal dynamics.

WildfireConvLSTM
  Two stacked ConvLSTM layers → Conv2d head → sigmoid output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell operating on 2-D spatial feature maps."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2

        # Combined gates: input (i), forget (f), cell (g), output (o)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,                        # (B, C_in, H, W)
        h: torch.Tensor,                        # (B, C_h, H, W)
        c: torch.Tensor,                        # (B, C_h, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:     # (h_next, c_next)
        combined = torch.cat([x, h], dim=1)     # (B, C_in+C_h, H, W)
        gates = self.conv(combined)             # (B, 4*C_h, H, W)

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """
    Stacked ConvLSTM — processes a sequence of spatial frames and returns
    the final hidden state of the last layer.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: list[int],
        kernel_size: int = 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = input_channels
        for hid_ch in hidden_channels:
            self.layers.append(ConvLSTMCell(in_ch, hid_ch, kernel_size))
            in_ch = hid_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C, H, W)

        Returns
        -------
        Tensor : (B, C_last_hidden, H, W)  — hidden state after last timestep
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Initialise hidden/cell states for each layer
        states = [
            layer.init_hidden(B, H, W, device)
            for layer in self.layers
        ]

        # Unroll through time
        layer_input = x  # (B, T, C, H, W)
        for layer_idx, layer in enumerate(self.layers):
            h, c = states[layer_idx]
            outputs = []
            for t in range(T):
                h, c = layer(layer_input[:, t], h, c)
                outputs.append(h)
            # Pass this layer's outputs as input to the next layer
            layer_input = torch.stack(outputs, dim=1)  # (B, T, C_h, H, W)

        return h  # final hidden state of last layer: (B, C_last_hidden, H, W)


class WildfireConvLSTM(nn.Module):
    """
    Full wildfire ignition prediction model.

    Encoder : ConvLSTM(input_channels, [64, 32])
    Head    : Conv2d(32→16, 3×3) → BN → ReLU → Conv2d(16→1, 1×1) → Sigmoid
    """

    def __init__(self, input_channels: int = 5, hidden_channels: list[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 32]

        self.encoder = ConvLSTM(input_channels, hidden_channels, kernel_size=3)

        last_hidden = hidden_channels[-1]
        self.head = nn.Sequential(
            nn.Conv2d(last_hidden, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C, H, W)

        Returns
        -------
        Tensor : (B, 1, H, W)  — per-pixel ignition probability
        """
        h = self.encoder(x)   # (B, C_last_hidden, H, W)
        return self.head(h)   # (B, 1, H, W)
