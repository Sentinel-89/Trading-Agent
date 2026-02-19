
"""backend/rl/trading_env_continuous.py

Continuous-action trading environment for portfolio allocation.

- Action: unbounded logits -> softmax -> long-only weights (optionally with cash)
- Optional top-k projection across assets (cash is never dropped)
- Reward: portfolio log-return (default) after turnover costs

This env uses the same frozen GRU encoder as the discrete env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch

from backend.models.gru_encoder import GRUEncoder


@dataclass
class TradingEnvContinuousConfig:
    # --------------------------------------------------------
    # Episode slicing
    # --------------------------------------------------------
    # NOTE: Despite the name, this is a *warmup/burn-in* constraint (min start index).
    # Set to 0 to disable feature burn-in; GRU still enforces `gru_window - 1`.
    window_length: int = 0

    # Number of tradable steps in an episode (training mode). If <= 0, run as long as possible.
    episode_length: int = 256

    # If True, evaluation runs deterministically from earliest valid index until the end (or end_t).
    eval_deterministic: bool = False

    # --------------------------------------------------------
    # Portfolio definition
    # --------------------------------------------------------
    num_assets: int = 14
    include_cash: bool = True
    top_k: int = 0  # 0 => disabled; else keep only top_k assets (cash kept)

    # --------------------------------------------------------
    # Costs (applied on turnover)
    # --------------------------------------------------------
    transaction_cost: float = 0.0005  # 5 bps per unit turnover
    slippage: float = 0.0

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------
    reward_mode: str = "log_return"  # "log_return" or "pnl"

    # --------------------------------------------------------
    # Extra regularization
    # --------------------------------------------------------
    turnover_penalty: float = 0.0005


class TradingEnvContinuous(gym.Env):
    """Continuous portfolio allocation environment (long-only)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_by_symbol: Dict[str, Any],
        feature_cols: List[str],
        config: Optional[TradingEnvContinuousConfig] = None,
        seed: Optional[int] = None,
        encoder_ckpt_path: str = "",
        device: str = "cpu",
        gru_window: int = 30,
        gru_hidden_size: int = 64,
    ):
        super().__init__()
        # Initialize market universe, frozen encoder, spaces, and episode state.

        self.data_by_symbol = data_by_symbol
        self.feature_cols = feature_cols
        self.cfg = config or TradingEnvContinuousConfig()

        self._rng = np.random.default_rng(seed)

        # Frozen GRU encoder (same as discrete env)
        self.device = str(device)
        self.gru_window = int(gru_window)
        self.latent_dim = int(gru_hidden_size)

        if not encoder_ckpt_path:
            raise ValueError("encoder_ckpt_path must be provided: continuous env requires frozen GRU")

        self.encoder = self._load_frozen_encoder(
            ckpt_path=str(encoder_ckpt_path),
            input_size=len(self.feature_cols),
            hidden_size=self.latent_dim,
        )

        # Action space: logits -> softmax -> weights
        self.num_assets = int(self.cfg.num_assets)
        self.include_cash = bool(self.cfg.include_cash)
        self.asset_dim = self.num_assets
        self.action_dim = self.asset_dim + (1 if self.include_cash else 0)

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # Observation: [per-asset GRU latent] + [current weights] + [equity]
        obs_dim = (self.asset_dim * self.latent_dim) + self.action_dim + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Episode state
        self.symbols: List[str] = []
        self.dfs: List[Any] = []
        self.t: int = 0
        self.start_t: int = 0
        self.end_t: int = 0  # exclusive

        # Portfolio state
        self.weights: np.ndarray = np.zeros((self.action_dim,), dtype=np.float32)
        if self.include_cash:
            self.weights[-1] = 1.0
        else:
            self.weights[: self.asset_dim] = 1.0 / float(self.asset_dim)

        self.equity: float = 1.0

        # Return tracking
        self.prev_close: np.ndarray = np.zeros((self.asset_dim,), dtype=np.float32)
        self.price_cols: List[str] = ["Close"] * self.asset_dim

        # Diagnostics
        self.turnover: float = 0.0
        self.rebalances: int = 0

        # Runtime controls (can be overridden per-reset via options)
        self._top_k_runtime: int = int(self.cfg.top_k)
        self._rebalance_every_n: int = 1
        self._steps_since_reset: int = 0

    # ============================================================
    # Init helpers
    # ============================================================

    def _load_frozen_encoder(self, ckpt_path: str, input_size: int, hidden_size: int) -> GRUEncoder:
        # Load encoder weights once and disable gradient updates.
        enc = GRUEncoder(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=1,
            dropout=0.0,
        ).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device)
        enc.load_state_dict(state)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        return enc

    # ============================================================
    # Helpers
    # ============================================================

    def _ones_mask(self) -> np.ndarray:
        # Continuous env: no invalid actions (all ones)
        return np.ones((self.action_dim,), dtype=np.float32)

    def _select_symbols(self, options: Optional[dict]) -> None:
        """Pick the symbols to trade for this episode."""
        # Resolve basket symbols and validate required data columns.
        all_syms = list(self.data_by_symbol.keys())
        if not all_syms:
            raise ValueError("data_by_symbol is empty")

        if options and options.get("symbols") is not None:
            syms = list(options["symbols"])
        else:
            if len(all_syms) < self.num_assets:
                raise ValueError(f"Need at least {self.num_assets} symbols, got {len(all_syms)}")
            syms = list(self._rng.choice(all_syms, size=self.num_assets, replace=False))

        if len(syms) != self.num_assets:
            raise ValueError(f"Expected exactly num_assets={self.num_assets} symbols, got {len(syms)}")

        dfs: List[Any] = []
        for sym in syms:
            if sym not in self.data_by_symbol:
                raise KeyError(f"Unknown symbol: {sym}")

            df = self.data_by_symbol[sym]

            missing = [c for c in self.feature_cols if c not in df.columns]
            if "Close" not in df.columns and "Close_raw" not in df.columns:
                missing.append("Close/Close_raw")
            if missing:
                raise ValueError(f"Symbol {sym}: missing columns: {missing}")

            dfs.append(df)

        self.symbols = syms
        self.dfs = dfs

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        x = x - float(np.max(x))
        ex = np.exp(x)
        s = float(np.sum(ex))
        if s <= 0.0:
            return np.ones_like(x, dtype=np.float32) / float(len(x))
        return (ex / s).astype(np.float32, copy=False)

    def _project_top_k_assets(self, w: np.ndarray) -> np.ndarray:
        """Keep only top-k asset weights (cash is never dropped), then renormalize."""
        k = int(self._top_k_runtime)
        if k <= 0:
            return w

        k = min(k, self.asset_dim)
        w = w.copy()

        asset_w = w[: self.asset_dim]
        cash_w = w[self.asset_dim :]  # [] or [cash]

        if k < self.asset_dim:
            idx = np.argpartition(asset_w, -k)[-k:]
            mask = np.zeros_like(asset_w)
            mask[idx] = 1.0
            asset_w = asset_w * mask

        total = float(np.sum(asset_w) + np.sum(cash_w))
        if total <= 0.0:
            w[:] = 0.0
            if self.include_cash:
                w[-1] = 1.0
            else:
                w[: self.asset_dim] = 1.0 / float(self.asset_dim)
            return w

        w[: self.asset_dim] = asset_w / total
        if self.include_cash:
            w[-1] = float(cash_w[0] / total)
        return w

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Convert raw logits -> valid long-only weights."""
        # Map unconstrained logits to normalized portfolio weights.
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {a.shape[0]}")

        w = self._softmax(a)
        w = self._project_top_k_assets(w)

        # Safety: clean up numeric issues and re-normalize
        w = np.clip(w, 0.0, 1.0)
        s = float(np.sum(w))
        if s <= 0.0:
            w[:] = 0.0
            if self.include_cash:
                w[-1] = 1.0
            else:
                w[: self.asset_dim] = 1.0 / float(self.asset_dim)
        else:
            w /= s

        return w.astype(np.float32, copy=False)

    def _get_obs(self) -> np.ndarray:
        """Observation = [GRU latent per asset] + [current weights] + [equity]."""
        # Build observation from per-asset latent features and portfolio state.
        feats: List[np.ndarray] = []

        w = int(self.gru_window)
        start = max(0, int(self.t - (w - 1)))

        for df in self.dfs:
            window = df.iloc[start : self.t + 1][self.feature_cols].to_numpy(dtype=np.float32, copy=False)
            x = torch.tensor(window, device=self.device).unsqueeze(0)
            with torch.no_grad():
                latent = self.encoder(x).squeeze(0)
            feats.append(latent.detach().cpu().numpy().astype(np.float32, copy=False))

        feats_flat = np.concatenate(feats, axis=0)
        return np.concatenate(
            [
                feats_flat,
                self.weights.astype(np.float32, copy=False),
                np.array([self.equity], dtype=np.float32),
            ],
            axis=0,
        )

    def _burn_in(self, n_min: int) -> int:
        """Minimum valid start index (warmup)."""
        warmup_cfg = int(self.cfg.window_length)
        warmup = max(warmup_cfg, int(self.gru_window) - 1)
        # Must allow at least one step forward
        return int(min(warmup, n_min - 2))

    # ============================================================
    # Core Gym API
    # ============================================================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Configure episode range, reset portfolio state, and return initial observation.
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._select_symbols(options)

        lengths = [len(df) for df in self.dfs]
        n_min = int(min(lengths))
        if n_min < 3:
            raise ValueError(f"Dataframe too short for trading: min length across symbols is {n_min}")

        min_t = self._burn_in(n_min)

        # Options
        mode = options.get("mode") if options else None
        is_eval = (mode == "eval") or bool(self.cfg.eval_deterministic)

        # Evaluation-time overrides (safe: only affect this episode)
        # - top_k: force sparsity during eval without retraining
        # - rebalance_every_n: apply new weights only every N steps (1=daily, 5=weekly)
        if options and options.get("top_k") is not None:
            self._top_k_runtime = int(options["top_k"])
        else:
            self._top_k_runtime = int(self.cfg.top_k)

        if options and options.get("rebalance_every_n") is not None:
            self._rebalance_every_n = max(1, int(options["rebalance_every_n"]))
        else:
            self._rebalance_every_n = 1

        self._steps_since_reset = 0

        episode_len_opt = int(options["episode_length"]) if (options and options.get("episode_length") is not None) else None
        start_t_opt = int(options["start_t"]) if (options and options.get("start_t") is not None) else None
        end_t_opt = int(options["end_t"]) if (options and options.get("end_t") is not None) else None  # inclusive

        if is_eval:
            # Default eval: run from earliest valid index through end of data.
            self.start_t = min_t if start_t_opt is None else start_t_opt
            self.start_t = int(np.clip(self.start_t, min_t, n_min - 2))

            if end_t_opt is not None:
                end_inclusive = int(np.clip(end_t_opt, self.start_t + 1, n_min - 1))
                self.end_t = end_inclusive + 1
            else:
                self.end_t = n_min

        else:
            # Training: rolling episodes
            episode_len = int(self.cfg.episode_length) if self.cfg.episode_length is not None else 0
            if episode_len_opt is not None:
                episode_len = int(episode_len_opt)

            if episode_len <= 0:
                episode_len = n_min - min_t - 1
            episode_len = max(2, int(episode_len))

            if start_t_opt is not None:
                self.start_t = int(start_t_opt)
            else:
                max_start = n_min - episode_len - 1
                if max_start <= min_t:
                    self.start_t = min_t
                else:
                    self.start_t = int(self._rng.integers(low=min_t, high=max_start))

            self.start_t = int(np.clip(self.start_t, min_t, n_min - 2))

            if end_t_opt is not None:
                end_inclusive = int(np.clip(end_t_opt, self.start_t + 1, n_min - 1))
                self.end_t = end_inclusive + 1
            else:
                self.end_t = self.start_t + episode_len

        # Final safety clip
        self.end_t = int(np.clip(self.end_t, self.start_t + 2, n_min))
        if self.end_t - self.start_t < 2:
            raise ValueError(
                f"Episode too short after slicing: start_t={self.start_t}, end_t(excl)={self.end_t}, n_min={n_min}"
            )

        self.t = self.start_t

        # Portfolio reset
        self.weights[:] = 0.0
        if self.include_cash:
            self.weights[-1] = 1.0
        else:
            self.weights[: self.asset_dim] = 1.0 / float(self.asset_dim)

        self.equity = 1.0

        # Price tracking (prefer raw prices if provided)
        self.price_cols = []
        for i, df in enumerate(self.dfs):
            pc = "Close_raw" if "Close_raw" in df.columns else "Close"
            self.price_cols.append(pc)
            self.prev_close[i] = float(df.iloc[self.t][pc])

        self.turnover = 0.0
        self.rebalances = 0

        obs = self._get_obs()
        info = {
            "symbols": list(self.symbols),
            "mode": str(mode) if mode is not None else ("eval" if is_eval else "train"),
            "min_t": int(min_t),
            "step": int(self.t),
            "start_t": int(self.start_t),
            "end_t": int(self.end_t),
            "episode_length": int(self.end_t - self.start_t),
            "top_k": int(self._top_k_runtime),
            "rebalance_every_n": int(self._rebalance_every_n),
            "equity": float(self.equity),
            "weights": self.weights.copy(),
            "turnover": float(self.turnover),
            "action_mask": self._ones_mask(),
            "price_cols": list(self.price_cols),
        }
        return obs, info

    def step(self, action):
        # Apply action weights, update portfolio equity, and advance one timestep.
        # Convert logits to valid target weights.
        # For rebalance_every_n > 1, portfolio updates occur only on rebalance steps.
        # Non-rebalance steps keep previous weights and avoid turnover costs.
        do_rebalance = (self._steps_since_reset % self._rebalance_every_n) == 0
        if do_rebalance:
            w_target = self._action_to_weights(action)
        else:
            w_target = self.weights

        # Turnover and costs (L1 change in weights)
        delta = w_target - self.weights
        turnover = float(np.sum(np.abs(delta)))
        cost = turnover * float(self.cfg.transaction_cost + self.cfg.slippage)
        if turnover > 1e-8:
            self.rebalances += 1

        # Advance time (apply weights to the return from prev_close -> current close)
        self.t += 1
        if self.t >= self.end_t:
            self.t = self.end_t - 1

        # Compute per-asset returns
        rets = np.zeros((self.asset_dim,), dtype=np.float32)
        closes_used: List[float] = []
        closes_raw: List[Optional[float]] = []

        for i, df in enumerate(self.dfs):
            pc = self.price_cols[i] if i < len(self.price_cols) else (
                "Close_raw" if "Close_raw" in df.columns else "Close"
            )
            close = float(df.iloc[self.t][pc])
            rets[i] = (close / float(self.prev_close[i])) - 1.0
            self.prev_close[i] = close

            closes_used.append(close)
            closes_raw.append(float(df.iloc[self.t]["Close_raw"]) if "Close_raw" in df.columns else None)

        # Cash earns 0
        port_ret = float(np.sum(w_target[: self.asset_dim] * rets))

        step_return = port_ret - cost
        step_return -= float(self.cfg.turnover_penalty) * turnover

        # Reward + equity update
        step_return_safe = max(float(step_return), -0.999999)
        if self.cfg.reward_mode == "log_return":
            reward = float(np.log1p(step_return_safe))
            self.equity *= float(np.exp(reward))
        else:
            reward = float(step_return_safe)
            self.equity *= float(1.0 + step_return_safe)

        # Commit
        self.weights = w_target
        self.turnover += turnover
        self._steps_since_reset += 1

        terminated = False
        truncated = self.t >= (self.end_t - 1)

        obs = self._get_obs()
        info = {
            "symbols": list(self.symbols),
            "step": int(self.t),
            "start_t": int(self.start_t),
            "end_t": int(self.end_t),
            "episode_length": int(self.end_t - self.start_t),
            "top_k": int(self._top_k_runtime),
            "rebalance_every_n": int(self._rebalance_every_n),
            "did_rebalance": bool(do_rebalance),
            "equity": float(self.equity),
            "weights": self.weights.copy(),
            "turnover": float(self.turnover),
            "rebalances": int(self.rebalances),
            "step_portfolio_return": float(port_ret),
            "step_cost": float(cost),
            "action_mask": self._ones_mask(),
            "close_used": closes_used,
            "close_raw": closes_raw,
        }

        return obs, reward, terminated, truncated, info
