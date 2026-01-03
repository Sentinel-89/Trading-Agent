"""
Main FastAPI Application
==========================================
This module defines the HTTP API layer of the Trading Agent backend.

IMPORTANT CONCEPTUAL NOTES
--------------------------

This file is intentionally kept *thin*:

- It does NOT contain:
    • Kite authentication logic
    • Instrument map loading
    • Historical data ingestion business logic
    • Technical feature computation
    • Labeling or ML preprocessing

All of these responsibilities are delegated to dedicated service modules
so that:
    • each module has a single responsibility,
    • the overall system becomes easier to maintain,
    • model training code can reuse ingestion/feature logic without importing FastAPI,
    • your codebase scales cleanly as RL / GRU model components are added.

This structure mirrors professional ML/quant backend architectures.

main.py therefore acts strictly as:
    - the FastAPI setup layer,
    - the routing layer,
    - the communication interface between the frontend and backend,
    - a small coordinator of the underlying service modules.
"""

from dotenv import load_dotenv
load_dotenv()  # Load Kite API credentials (API key, access token) from .env at startup.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# The API does not compute features or fetch data itself.  
# It delegates to service modules:
from backend.services.kite_service import fetch_historical_data
from backend.features.technical_features import calculate_technical_features

import pandas as pd


# =====================================================================================
# FastAPI Initialization
# =====================================================================================

app = FastAPI(title="Trading Agent Backend")

"""
CORS CONFIGURATION
-------------------

The FastAPI backend and Next.js frontend run on different localhost ports:
    - Backend: 8000
    - Frontend: 3000

The browser enforces the Same-Origin Policy, which would normally block
cross-origin requests.

CORS middleware explicitly permits the frontend to communicate with the backend.
This is required for local development and container setups.
"""

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # frontend URLs allowed to call this backend
    allow_credentials=True,
    allow_methods=["*"],         # allow GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],         # allow all custom headers
)


# =====================================================================================
# API ENDPOINTS
# =====================================================================================

@app.get("/")
def read_root():
    """
    Simple health-check endpoint.
    Allows frontend / monitoring tools to confirm backend is alive.
    """
    return {"message": "Trading Agent Backend is running!"}


@app.get("/api/v1/features/{symbol}")
def get_features(symbol: str, start_date: str = "2020-01-01"):
    """
    CORE FRONTEND ENDPOINT
    ----------------------

    This endpoint powers your Next.js dashboard chart.

    Pipeline:
        1. Fetch raw OHLCV from Kite (via kite_service)
        2. Compute standardized technical features (via technical_features.py)
        3. Convert to JSON-friendly format
        4. Return the most recent slice (default: last 100 rows = days) for frontend-visualization

    The frontend expects a list of objects shaped like:

        {
            "Date": "2021-01-15",
            "Open": ...,
            "Close": ...,
            "RSI": ...,
            ...
        }

    where:
        - Date is string-serialized (YYYY-MM-DD)
        - All numbers are float64 (already ensured by feature pipeline)
        - The frontend adds simulated Buy/Sell markers for visualization only

    This endpoint is deliberately *stateless*:  
    it does NOT retain or update any trading context.  
    It simply exposes data for visualization.
    """

    # 1. Pull raw data from Zerodha Kite
    raw_df = fetch_historical_data(symbol, start_date)
    if raw_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch or generate data for {symbol}. "
                   "Check your API connection or symbol spelling."
        )

    # 2. Compute features
    feature_df = calculate_technical_features(raw_df)
    if feature_df.empty:
        raise HTTPException(
            status_code=400,
            detail="Feature calculation failed (insufficient data remaining after lookbacks)."
        )

    # 3. Prepare JSON output
    feature_df = feature_df.copy()
    feature_df.reset_index(inplace=True)

    # Date serialization (pandas Timestamp → str)
    feature_df["Date"] = feature_df["Date"].dt.strftime("%Y-%m-%d")

    # Whitelist columns returned to frontend
    # NOTE: RealizedVol_20 is intentionally EXCLUDED from frontend visualization.
    #       It is used for ML training only, not chart UI display.
    feature_columns_frontend = [
        "Date", "Open", "Close",
        "RSI", "MACD", "MACD_Signal", "ATR",
        "SMA_50", "OBV", "ROC_10", "SMA_Ratio"
    ]

    # Return the last 100 days of windowed features
    #The reason for the 100-day return is purely for visualization, whereas the 30-day window is for computation.
    return {
        "symbol": symbol,
        "features": feature_df[feature_columns_frontend].tail(100).to_dict("records")
    }

    # features is a list of objects where each object contains Date as a string (YYYY-MM-DD) and numeric fields. The CharingDashboard.tsx component’s FeatureData TypeScript interface matches that exactly!