# backend/models/models.py
"""
Neural Network Architectures
===============================================================
This module defines the GRU-based encoder that forms the backbone
of your trading agent.

The encoder is used in two distinct training phases and This file will eventually be used by:

    • train_gru.py (supervised training stage)
    • rl_agent.py (encoder reused + frozen or fine-tuned)

----------------------------------------------------------------
1. SUPERVISED PRETRAINING (1st STAGE)
----------------------------------------------------------------
Goal:
    • Teach the GRU to extract meaningful temporal signals:
        - trend direction
        - expected return magnitude

How:
    • The GRU consumes sliding windows of engineered features
    • At each window, two prediction heads are used:
        - Trend classification head (direction = -1 / 0 / +1)
        - Expected-return regression head (magnitude over horizon)

Why:
    • These tasks bootstrap the GRU into understanding structures:
        - momentum vs reversal
        - magnitude of future moves
        - market context encoded in technical indicators

You will train:
    • Trend: CrossEntropyLoss
    •Expected return: MSELoss or HuberLoss

Total loss (example):
    total_loss = CE(trend_logits, trend_labels)  
            + λ * MSE(return_pred, expected_return)
A typical λ is 1.0 or 0.5.

----------------------------------------------------------------
2. RL TRAINING (LATER)
----------------------------------------------------------------
Goal:
    • Use the pretrained GRU as the "state encoder" for the RL agent.
    • The Actor-Critic (PPO / DQN / SAC) sees only:
          latent_state = GRU(last_window)
    • Pretraining dramatically speeds up learning and stabilizes RL.

During RL:

    • You load the pretrained model
    • Extract only:
      encoder = trained_model.encoder
    • Use output latent vector h as state representation

----------------------------------------------------------------
IMPORTANT DESIGN CHOICES
----------------------------------------------------------------
• GRU is used because financial time series have long dependencies
  but are noisy — GRUs strike a balance between capacity and stability.

• We produce only ONE hidden state per input window (the final h_t),
  which becomes the *latent representation* for classification,
  regression, or RL policy/value networks.

• Trend classification uses 3 logits (down, flat, up)
    → training uses CrossEntropyLoss

• Expected return uses a scalar regression head
    → training uses MSELoss or HuberLoss

"""

import torch
import torch.nn as nn
from typing import Optional


# =====================================================================
# GRU Encoder
# =====================================================================

class GRUEncoder(nn.Module):
    """
    GRU-based temporal feature extractor.
    
    Input:
        X: Tensor of shape (Batch, SeqLen, NumFeatures)

    Output:
        h_final: Tensor of shape (Batch, HiddenSize)

    This module extracts latent representations from historical windows.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # We use batch_first=True so input is (B, T, F)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence and returns ONLY the final hidden state.
        """
        # GRU returns:
        #   output: (B, T, H)
        #   h_n:    (num_layers, B, H)
        output, h_n = self.gru(x)

        # We return the final hidden state from the top layer: (B, H)
        h_final = h_n[-1]
        return h_final


# =====================================================================
# Multi-Task Supervised Heads
# =====================================================================

class TrendHead(nn.Module):
    """
    Classification head:
        Predicts -1 / 0 / +1 → implemented as classes {0,1,2}

    Output:
        logits: (Batch, 3)
    
    Loss function used later:
        CrossEntropyLoss
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)  # no softmax — handled by loss


class ReturnHead(nn.Module):
    """
    Regression head:
        Predicts expected return over a fixed horizon (e.g., 3 days)

    Output:
        scalar value per sample → (Batch, 1)

    Loss:
        MSELoss or HuberLoss (recommended)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


# =====================================================================
# Combined Supervised Model
# =====================================================================

class GRUSupervisedModel(nn.Module):
    """
    Full pretraining model:
        GRU Encoder + Trend Classification Head + Expected Return Head

    During training:
        - compute trend_logits for CrossEntropyLoss
        - compute expected_return_pred for MSELoss

    This model is used ONLY in supervised pretraining.
    During RL training, we will use ONLY self.encoder to generate
    latent states for the actor and critic networks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Shared GRU encoder
        self.encoder = GRUEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Two supervised heads
        self.trend_head = TrendHead(hidden_size)
        self.return_head = ReturnHead(hidden_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass:

            x → GRUEncoder → h
            → TrendHead(h)
            → ReturnHead(h)

        Returns:
            trend_logits (B, 3)
            expected_return_pred (B, 1)
        """
        h = self.encoder(x)

        trend_logits = self.trend_head(h)
        expected_return_pred = self.return_head(h)

        return trend_logits, expected_return_pred
