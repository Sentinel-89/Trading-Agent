# ============================================================
#
# TradingEnv v1 / v2 / v3 / v4 / v5
#
# Gymnasium-compatible trading environment for PPO.
#

# ============================================================

from typing import Dict, Tuple, Optional, Any

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

    v4 (log-only / shaped):
      - Observation: v3 +
                     equity (normalized)
                     drawdown (normalized)   (ℝ⁶⁹)

    v5 (training-only):
      - Same as v4 observations
      - Small unrealized reward shaping while holding

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
        entry_transaction_cost: Optional[float] = None,  # if None, use transaction_cost
        holding_penalty: float = 0.0,  # per-step penalty while holding a position
        device: str = "cpu",
        max_holding_period: int = 252,   
        initial_cash: float = 1.0,       # used in v4/v5 only (normalized)
        use_action_mask: bool = True,    # when True, info["action_mask"] reflects valid actions
        invalid_action_penalty: float = 0.001,  # optional penalty when use_action_mask=False
    ):
        super().__init__()

        # --------------------------------------------------------
        # Environment version guard (Axis A)
        # --------------------------------------------------------
        assert env_version in ("v1", "v2", "v3", "v4", "v5")
        self.env_version = env_version

        # --------------------------------------------------------
        # v4 / v5 reward shaping configuration
        # --------------------------------------------------------
        # v4: (optional) drawdown shaping on trade close
        # v5: drawdown shaping + small unrealized shaping while holding
        self.enable_drawdown_shaping = env_version in ("v4", "v5")
        self.enable_unrealized_shaping = env_version == "v5"

        # Tunables
        self.lambda_drawdown = 0.002
        self.lambda_unrealized = 0.0001

        # --------------------------------------------------------
        # Episode construction mode (Axis B)
        # --------------------------------------------------------
        assert episode_mode in ("full_history", "rolling_window")
        self.episode_mode = episode_mode
        self.window_length = window_length
        self.random_start = random_start

        if self.episode_mode == "rolling_window":
            assert self.window_length is not None

        self.transaction_cost = transaction_cost
        self.entry_transaction_cost = float(
            transaction_cost if entry_transaction_cost is None else entry_transaction_cost
        )
        self.holding_penalty = float(holding_penalty)
        self.device = device
        self.use_action_mask = bool(use_action_mask)
        self.invalid_action_penalty = float(invalid_action_penalty)

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
        elif self.env_version == "v3":
            obs_dim = 67
        else:  # v4 / v5
            obs_dim = 69

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)

        # --------------------------------------------------------
        # Episode / Position State
        # --------------------------------------------------------
        self.current_symbol: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None

        self.current_step: int = 0
        self.last_step: int = 0

        self.position: int = 0
        self.entry_price: Optional[float] = None
        self.entry_step: Optional[int] = None

        # --------------------------------------------------------
        # portfolio state
        # --------------------------------------------------------
        self.initial_cash = initial_cash
        self.cash: float = initial_cash
        self.equity: float = initial_cash
        self.equity_peak: float = initial_cash

        # --------------------------------------------------------
        # per-trade risk tracking (internal)
        # --------------------------------------------------------
        self.max_drawdown_in_trade: float = 0.0
        self.trade_peak_equity: Optional[float] = None

        self.max_holding_period = max_holding_period

        # Diagnostics (initialized on reset)
        self.executed_buys = 0
        self.executed_sells = 0
        self.invalid_actions = 0

    # ============================================================
    # Core Gym API
    # ============================================================

    def get_true_action_mask(self) -> np.ndarray:
        """Return the always-correct 0/1 mask over actions [HOLD, BUY, SELL]."""
        mask = np.array([1, 1, 1], dtype=np.uint8)
        if self.position == 0:
            mask[2] = 0  # SELL invalid when flat
        elif self.position == 1:
            mask[1] = 0  # BUY invalid when long
        return mask

    def get_action_mask(self) -> np.ndarray:
        """Return a 0/1 mask over actions [HOLD, BUY, SELL].

        - If `self.use_action_mask` is True: return the true constraint mask.
        - If `self.use_action_mask` is False: return all-ones (mask ablation).
        """
        if not self.use_action_mask:
            return np.array([1, 1, 1], dtype=np.uint8)
        return self.get_true_action_mask()

    def reset(
        self,
        *,
        seed=None,
        options: Optional[Any] = None,
        symbol: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:

        super().reset(seed=seed)

        # --------------------------------------------------------
        # Symbol selection
        # - single-env: allow `symbol=`
        # - vector-env: allow `options={"symbol": ...}` or `options=[{"symbol": ...}]`
        # --------------------------------------------------------
        opt_symbol = None
        if symbol is not None:
            opt_symbol = symbol
        elif options is not None:
            if isinstance(options, dict):
                opt_symbol = options.get("symbol")
            elif isinstance(options, (list, tuple)) and len(options) > 0 and isinstance(options[0], dict):
                opt_symbol = options[0].get("symbol")

        if opt_symbol is None:
            self.current_symbol = str(self.np_random.choice(self.symbols))
        else:
            assert opt_symbol in self.data_by_symbol, f"Unknown symbol: {opt_symbol}"
            self.current_symbol = str(opt_symbol)

        self.df = self.data_by_symbol[self.current_symbol].reset_index(drop=True)

        # Need at least 30 bars for the GRU window (0..29)
        min_start = 29
        max_step = int(len(self.df) - 1)
        if max_step < min_start:
            raise ValueError(
                f"Not enough rows for symbol={self.current_symbol}. "
                f"Need >= {min_start + 1} rows, got {len(self.df)}."
            )

        # Episode construction
        # --------------------------------------------------------
        if self.episode_mode == "full_history":
            start_step = min_start
            last_step = max_step
        else:
            assert self.window_length is not None
            max_start = max_step - int(self.window_length) + 1
            if max_start < min_start:
                raise ValueError(
                    f"window_length={self.window_length} too large for symbol={self.current_symbol} "
                    f"(rows={len(self.df)})."
                )
            if self.random_start:
                start_step = int(self.np_random.integers(min_start, max_start + 1))
            else:
                start_step = min_start
            last_step = int(start_step + int(self.window_length) - 1)

        self.current_step = int(start_step)
        self.last_step = int(last_step)

        # --------------------------------------------------------
        # Reset position state
        # --------------------------------------------------------
        self.position = 0
        self.entry_price = None
        self.entry_step = None

        # --------------------------------------------------------
        # Reset portfolio state (v4/v5)
        # --------------------------------------------------------
        self.cash = self.initial_cash
        self.equity = self.initial_cash
        self.equity_peak = self.initial_cash

        # --------------------------------------------------------
        # Reset per-trade risk state
        # --------------------------------------------------------
        self.max_drawdown_in_trade = 0.0
        self.trade_peak_equity = None

        # Diagnostics
        self.executed_buys = 0
        self.executed_sells = 0
        self.invalid_actions = 0

        obs = self._get_observation()

        info = {
            "symbol": self.current_symbol,
            "env_version": self.env_version,
            "episode_mode": self.episode_mode,
            "action_mask": self.get_action_mask(),
            "true_action_mask": self.get_true_action_mask(),
            "use_action_mask": bool(self.use_action_mask),
            "entry_transaction_cost": float(self.entry_transaction_cost),
            "holding_penalty": float(self.holding_penalty),
        }

        return obs, info

    def step(self, action: int):
        assert self.df is not None

        reward = 0.0
        terminated = False
        truncated = False

        # True validity w.r.t. the environment constraints (independent of self.use_action_mask)
        invalid = False
        if self.position == 0 and action == 2:
            invalid = True  # SELL while flat
        elif self.position == 1 and action == 1:
            invalid = True  # BUY while long

        if invalid:
            self.invalid_actions += 1
            # Optional penalty for invalid actions when masking is disabled
            if (not self.use_action_mask) and self.invalid_action_penalty != 0.0:
                reward -= self.invalid_action_penalty

        curr_price = float(self.df.loc[self.current_step, "Close_raw"])

        # --------------------------------------------------------
        # Action semantics
        # --------------------------------------------------------
        # BUY
        if action == 1:
            if self.position == 0:
                # Charge transaction cost on entry
                self.cash *= (1.0 - self.entry_transaction_cost)

                self.position = 1
                self.entry_price = curr_price
                self.entry_step = self.current_step
                self.max_drawdown_in_trade = 0.0
                self.trade_peak_equity = self.equity
                self.executed_buys += 1

        # SELL
        elif action == 2:
            if self.position == 1:
                assert self.entry_price is not None

                raw_return = (curr_price - self.entry_price) / self.entry_price
                reward = raw_return - self.transaction_cost

                if self.enable_drawdown_shaping:
                    reward -= self.lambda_drawdown * abs(self.max_drawdown_in_trade)

                self.cash *= (1.0 + raw_return - self.transaction_cost)

                self.executed_sells += 1
                self.trade_peak_equity = None
                self.position = 0
                self.entry_price = None
                self.entry_step = None

        # --------------------------------------------------------
        # Portfolio update
        # --------------------------------------------------------
        if self.position == 1:
            assert self.entry_price is not None
            unrealized = (curr_price - self.entry_price) / self.entry_price
            self.equity = self.cash * (1.0 + unrealized)
        else:
            self.equity = self.cash

        self.equity_peak = max(self.equity_peak, self.equity)

        # --------------------------------------------------------
        # Update per-trade max drawdown
        # --------------------------------------------------------
        if self.position == 1:
            assert self.trade_peak_equity is not None
            self.trade_peak_equity = max(self.trade_peak_equity, self.equity)

            # drawdown is negative; more negative = worse
            drawdown = (
                (self.equity - self.trade_peak_equity) / self.trade_peak_equity
                if self.trade_peak_equity > 0.0
                else 0.0
            )
            self.max_drawdown_in_trade = min(self.max_drawdown_in_trade, drawdown)

        # --------------------------------------------------------
        # Small unrealized reward shaping (v5 only)
        # --------------------------------------------------------
        if self.enable_unrealized_shaping and self.position == 1:
            assert self.entry_price is not None
            unrealized = (curr_price - self.entry_price) / self.entry_price
            reward += self.lambda_unrealized * unrealized

        # --------------------------------------------------------
        # Optional per-step holding penalty (applies while long)
        # --------------------------------------------------------
        if self.position == 1 and self.holding_penalty != 0.0:
            reward -= self.holding_penalty

        # --------------------------------------------------------
        # Forced liquidation at episode end
        # --------------------------------------------------------
        forced_liquidation = False
        if self.position == 1 and self.current_step == self.last_step:
            assert self.entry_price is not None

            raw_return = (curr_price - self.entry_price) / self.entry_price
            self.cash *= (1.0 + raw_return - self.transaction_cost)

            # If the agent didn't SELL explicitly on this step, give terminal reward here.
            if action != 2:
                reward += raw_return - self.transaction_cost
                if self.enable_drawdown_shaping:
                    reward -= self.lambda_drawdown * abs(self.max_drawdown_in_trade)

            self.trade_peak_equity = None
            self.position = 0
            self.entry_price = None
            self.entry_step = None

            forced_liquidation = True
            self.executed_sells += 1

        # --------------------------------------------------------
        # Advance time (next-state) and handle truncation
        # --------------------------------------------------------
        self.current_step += 1
        if self.current_step > self.last_step:
            truncated = True

        obs = (
            self._get_observation()
            if not truncated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        info = {
            "symbol": self.current_symbol,
            "step": self.current_step,
            "position": self.position,
            "equity": self.equity,
            "action_mask": self.get_action_mask(),
            "true_action_mask": self.get_true_action_mask(),
            "executed_buys": int(self.executed_buys),
            "executed_sells": int(self.executed_sells),
            "invalid_actions": int(self.invalid_actions),
            "forced_liquidation": forced_liquidation,
        }

        return obs, reward, terminated, truncated, info

    # ============================================================
    # Observation Construction
    # ============================================================

    def _get_observation(self) -> np.ndarray:
        assert self.df is not None

        window = self.df.loc[
            self.current_step - 29 : self.current_step,
            self.feature_cols,
        ].values.astype(np.float32)

        x = torch.tensor(window, device=self.device).unsqueeze(0)

        with torch.no_grad():
            latent = self.encoder(x).squeeze(0)

        if self.env_version == "v1":
            return latent.cpu().numpy()

        position_flag = float(self.position)

        if self.env_version == "v2":
            return torch.cat([
                latent,
                torch.tensor([position_flag], device=latent.device),
            ]).cpu().numpy()

        # --------------------------------------------------------
        # v3+ extras
        # --------------------------------------------------------
        if self.position == 1:
            assert self.entry_step is not None
            assert self.entry_price is not None

            holding_time = self.current_step - self.entry_step
            time_in_trade = min(holding_time / self.max_holding_period, 1.0)

            curr_price = float(self.df.loc[self.current_step, "Close_raw"])
            unrealized = (curr_price - self.entry_price) / self.entry_price
        else:
            time_in_trade = 0.0
            unrealized = 0.0

        extras = [
            position_flag,
            float(time_in_trade),
            float(unrealized),
        ]

        # --------------------------------------------------------
        # v4 / v5: equity + drawdown
        # --------------------------------------------------------
        if self.env_version in ("v4", "v5"):
            if self.position == 1 and self.trade_peak_equity is not None:
                # drawdown is negative; more negative = worse
                drawdown = (self.equity - self.trade_peak_equity) / self.trade_peak_equity
            else:
                drawdown = 0.0
            extras.extend([
                float(self.equity),
                float(drawdown),
            ])

        obs = torch.cat([
            latent,
            torch.tensor(extras, device=latent.device),
        ])

        return obs.cpu().numpy()