"""
Main FastAPI Application (thin routing layer)
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import os

from backend.services.kite_service import fetch_historical_data
from backend.features.technical_features import calculate_technical_features

from backend.rl_inference import run_agent_from_kite_pad_mask
app = FastAPI(title="Trading Agent Backend")

# Allow localhost and common LAN development origins.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.0.6:3000",
]

# Allow additional 192.168.*.*:3000 origins for local-network development.
allow_origin_regex = r"^http://(localhost|127\\.0\\.0\\.1|192\\.168\\.\\d+\\.\\d+):3000$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Trading Agent Backend is running!"}


@app.get("/api/v1/features/{symbol}")
def get_features(symbol: str, start_date: str = "2020-01-01"):
    raw_df = fetch_historical_data(symbol, start_date)
    if raw_df is None or raw_df.empty:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")

    feature_df = calculate_technical_features(raw_df)
    if feature_df is None or feature_df.empty:
        raise HTTPException(status_code=400, detail="Feature calculation failed")

    feature_df = feature_df.copy()
    feature_df.reset_index(inplace=True)
    feature_df["Date"] = feature_df["Date"].dt.strftime("%Y-%m-%d")

    feature_columns_frontend = [
        "Date", "Open", "Close",
        "RSI", "MACD", "MACD_Signal", "ATR",
        "SMA_50", "OBV", "ROC_10", "SMA_Ratio", "RealizedVol_20",

    ]

    return {
        "symbol": symbol,
        "features": feature_df[feature_columns_frontend].tail(100).to_dict("records"),
    }


# ----------------------------
# RL RUN ENDPOINT (Kite live + pad+mask)
# ----------------------------

SCALER_PATH = os.getenv("SCALER_PATH", os.path.join("backend", "models", "checkpoints", "scaler.pkl"))
ENCODER_PATH = os.getenv("ENCODER_PATH", os.path.join("backend", "models", "checkpoints", "gru_encoder.pt"))
CKPT_PATH = os.getenv("CKPT_PATH", os.path.join("backend", "artifacts", "phase_d_continuous", "final.pt"))

FEATURE_COLS = os.getenv(
    "FEATURE_COLS",
    "Open,Close,RSI,MACD,MACD_Signal,ATR,SMA_50,OBV,ROC_10,SMA_Ratio,RealizedVol_20",
)

GRU_WINDOW = int(os.getenv("GRU_WINDOW", "30"))


class AgentRunRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=14)
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    rebalance: str = Field("monthly", description="daily|weekly|monthly")
    max_positions: int = Field(0, ge=0, le=14, description="0 => full 14, else top-k")
    rotation_factor: float = Field(0.0, ge=0.0, le=2.0, description="0 disables rotation bias")
    initial_cash: float = Field(100000.0, gt=0)


@app.post("/api/v1/agent/run")
def run_agent(req: AgentRunRequest):
    if req.start_date > req.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    feature_cols = [c.strip() for c in FEATURE_COLS.split(",") if c.strip()]

    return run_agent_from_kite_pad_mask(
        user_symbols=req.symbols,
        start_date=req.start_date,
        end_date=req.end_date,
        rebalance=req.rebalance,
        max_positions=req.max_positions,
        rotation_factor=req.rotation_factor,
        initial_cash=req.initial_cash,
        scaler_path=SCALER_PATH,
        encoder_path=ENCODER_PATH,
        ckpt_path=CKPT_PATH,
        feature_cols=feature_cols,
        num_assets_fixed=14,
        include_cash=True,
        gru_window=GRU_WINDOW,
    )
