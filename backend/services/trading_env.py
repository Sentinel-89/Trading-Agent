# ============================================================
# services/trading_env.py
#
# TradingEnv v1 / v2 / v3
#
# Gymnasium-compatible trading environment for PPO.
#
# Axis A: env_version (semantic meaning of observations/rewards)
# Axis B: episode_mode (how episodes are constructed)
# ============================================================

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

import torch

from backend.models.gru_encoder import GRUEncoder


class TradingEnv(gym.Env):
    """
    Versioned Trading Environment

    v1:
      - Observation: GRU latent state only (ℝ⁶⁴)

    v2:
      - Observation: latent + position flag (ℝ⁶⁵)

    v3:
      - Observation: latent + position flag
                     + normalized time-in-trade
                     + unrealized return (ℝ⁶⁷)

    Actions (all versions):
      - HOLD / BUY / SELL
      - Long-only, max one open position
      - Reward realized only on SELL
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        encoder_ckpt_path: str,
        feature_cols: list,
        env_version: str = "v1",
        episode_mode: str = "rolling_window",
        window_length: Optional[int] = 252,
        random_start: bool = True,
        transaction_cost: float = 0.001,
        device: str = "cpu",
        max_holding_period: int = 252,   # used in v3 only
    ):
        super().__init__()

        # --------------------------------------------------------
        # Environment version guard (Axis A)
        # --------------------------------------------------------
        assert env_version in ("v1", "v2", "v3")
        self.env_version = env_version

        # --------------------------------------------------------
        # Episode construction mode (Axis B)
        # --------------------------------------------------------
        assert episode_mode in ("full_history", "rolling_window")
        self.episode_mode = episode_mode
        self.window_length = window_length
        self.random_start = random_start

        if self.episode_mode == "rolling_window":
            assert (
                self.window_length is not None
            ), "window_length must be set for rolling_window episodes"

        self.transaction_cost = transaction_cost
        self.device = device

        # --------------------------------------------------------
        # Market data
        # --------------------------------------------------------
        self.data_by_symbol = data_by_symbol
        self.feature_cols = feature_cols
        self.symbols = list(data_by_symbol.keys())

        # --------------------------------------------------------
        # Frozen GRU Encoder
        # --------------------------------------------------------
        self.encoder = GRUEncoder(
            input_size=len(feature_cols),
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
        ).to(self.device)

        state = torch.load(encoder_ckpt_path, map_location=self.device)
        self.encoder.load_state_dict(state)
        self.encoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

        # --------------------------------------------------------
        # Gym Spaces
        # --------------------------------------------------------
        if self.env_version == "v1":
            obs_dim = 64
        elif self.env_version == "v2":
            obs_dim = 65
        else:  # v3
            obs_dim = 67

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Actions:
        #   0 -> HOLD
        #   1 -> BUY
        #   2 -> SELL
        self.action_space = spaces.Discrete(3)

        # --------------------------------------------------------
        # Episode / Position State
        #
        # NOTE:
        #   These variables are intentionally NOT part
        #   of the observation in v1.
        # --------------------------------------------------------
        self.current_symbol: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None

        # Episode boundaries
        self.current_step: int = 0
        self.last_step: int = 0

        # Position state
        self.position: int = 0          # 0 = flat, 1 = long
        self.entry_price: Optional[float] = None
        self.entry_step: Optional[int] = None

        # v3-only configuration
        self.max_holding_period = max_holding_period

    # ============================================================
    # Core Gym API
    # ============================================================

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Sample a symbol for this episode
        self.current_symbol = np.random.choice(self.symbols)
        self.df = self.data_by_symbol[self.current_symbol].reset_index(drop=True)

        # --------------------------------------------------------
        # Episode construction (Axis B)
        # --------------------------------------------------------
        min_start = 29  # encoder requires [t-29 : t]

        if self.episode_mode == "full_history":
            self.current_step = min_start
            self.last_step = len(self.df) - 1

        else:  # rolling_window
            assert self.window_length is not None
            max_start = len(self.df) - self.window_length - 1
            start = (
                self.np_random.integers(min_start, max_start)
                if self.random_start
                else min_start
            )
            self.current_step = int(start)
            self.last_step = int(start + self.window_length)

        # Reset position state
        self.position = 0
        self.entry_price = None
        self.entry_step = None

        obs = self._get_observation()

        info = {
            "symbol": self.current_symbol,
            "env_version": self.env_version,
            "episode_mode": self.episode_mode,
        }

        return obs, info

    def step(self, action: int):
        assert self.df is not None

        reward = 0.0
        terminated = False
        truncated = False

        close_price = float(self.df.loc[self.current_step, "Close"])

        # --------------------------------------------------------
        # Action semantics
        # --------------------------------------------------------
        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = close_price
                self.entry_step = self.current_step

        if action == 2:  # SELL
            if self.position == 1:
                assert self.entry_price is not None
                raw_return = (close_price - self.entry_price) / self.entry_price
                reward = raw_return - self.transaction_cost

                self.position = 0
                self.entry_price = None
                self.entry_step = None

        # --------------------------------------------------------
        # Advance time
        # --------------------------------------------------------
        self.current_step += 1

        # --------------------------------------------------------
        # Episode termination
        # --------------------------------------------------------
        if self.current_step >= self.last_step:
            terminated = True

            if self.position == 1:
                assert self.entry_price is not None
                final_price = float(self.df.loc[self.current_step, "Close"])
                raw_return = (final_price - self.entry_price) / self.entry_price
                reward += raw_return - self.transaction_cost

                self.position = 0
                self.entry_price = None
                self.entry_step = None

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        info = {
            "symbol": self.current_symbol,
            "step": self.current_step,
            "position": self.position,
        }

        return obs, reward, terminated, truncated, info

    # ============================================================
    # Helper Functions
    # ============================================================

    def _get_observation(self) -> np.ndarray:
        """
        Compute observation according to env_version.
        """
        assert self.df is not None

        # --------------------------------------------------------
        # Latent market state (all versions)
        # --------------------------------------------------------
        window = self.df.loc[
            self.current_step - 29 : self.current_step,
            self.feature_cols,
        ].values.astype(np.float32)

        x = torch.tensor(window, device=self.device).unsqueeze(0)

        with torch.no_grad():
            latent = self.encoder(x).squeeze(0)

        if self.env_version == "v1":
            return latent.cpu().numpy()

        # --------------------------------------------------------
        # v2+: position flag
        # --------------------------------------------------------
        position_flag = float(self.position)

        if self.env_version == "v2":
            obs = torch.cat([
                latent,
                torch.tensor([position_flag], device=latent.device),
            ])
            return obs.cpu().numpy()

        # --------------------------------------------------------
        # v3: time-in-trade + unrealized return
        # --------------------------------------------------------
        if self.position == 1:
            assert self.entry_price is not None
            assert self.entry_step is not None

            holding_time = self.current_step - self.entry_step
            time_in_trade = min(
                holding_time / self.max_holding_period,
                1.0,
            )

            close_price = float(self.df.loc[self.current_step, "Close"])
            unrealized_return = (
                (close_price - self.entry_price) / self.entry_price
            )
        else:
            time_in_trade = 0.0
            unrealized_return = 0.0

        obs = torch.cat([
            latent,
            torch.tensor(
                [
                    position_flag,
                    float(time_in_trade),
                    float(unrealized_return),
                ],
                device=latent.device,
            ),
        ])

        return obs.cpu().numpy()
