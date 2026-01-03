"""
Kite Connect Service Layer
===========================================================
This module encapsulates everything related to Zerodha Kite:

    • API key + access token initialization
    • Session creation and maintenance
    • Instrument master download (symbol → instrument_token)
    • Historical OHLCV fetching

I.e. the module is a single authoritative ingestion layer used by:
    • FastAPI backend
    • dataset generation scripts
    • labeling + preprocessing logic
    • future RL environments and GRU training

Separation Rationale
--------------------
Previously, main.py handled both API routing and business logic for:
    - authenticating Kite
    - loading the instrument master
    - querying historical data
    - error translation / schema normalization

This made main.py large and difficult to reuse.

The service module now provides a clean interface:

    from backend.services.kite_service import fetch_historical_data

This allows:
    • dataset generation scripts (generate_dataset.py)
    • model training scripts
    • CLI utilities
    • FastAPI routes

all to share the same ingestion logic without importing FastAPI.

This is the “single responsibility” pattern applied properly.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

from kiteconnect import KiteConnect as KiteApp   # Correct import alias

# Resolve the path to .env relative to this file
# Path(__file__) is /backend/services/kite_service.py
# .parent.parent moves up to /backend/
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# =====================================================================================
# GLOBAL SINGLETON STORAGE (Created at application startup)
# =====================================================================================

"""
KITE_CLIENT
-----------
A long-lived Zerodha Kite session used for all API calls.

This avoids:
    - re-authenticating on every request,
    - re-downloading the instrument master,
    - repeated session construction,
    - unnecessary network load.

INSTRUMENT_MAP
--------------
A cached DataFrame containing the entire F&O + equity universe available on NSE
(as returned by kite_client.instruments("NSE")).

Indexed by 'tradingsymbol':
    e.g., "TCS" → instrument_token=2953217

This lookup is required for historical_data() calls, which accept only tokens.
"""

KITE_CLIENT: Optional[KiteApp] = None
INSTRUMENT_MAP: Optional[pd.DataFrame] = None


# =====================================================================================
# INITIALIZATION (Executed on first import of this module)
# =====================================================================================

"""
This block mirrors the original main.py startup logic,
but is now located here — the correct place, because ONLY this module
is responsible for interacting with Kite.

main.py simply imports this module.  
FastAPI does not need to know *how* the client is created.
"""


def _initialize_kite_client():
    """Internal helper. Creates the Kite client and loads the instrument map."""
    global KITE_CLIENT, INSTRUMENT_MAP

    # Load environment variables
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if api_key:
        api_key = api_key.strip()  # defensive sanitization
    if access_token:
        access_token = access_token.strip()

    if not api_key or not access_token:
        raise ValueError(
            "FATAL: KITE_API_KEY or KITE_ACCESS_TOKEN missing. "
            "Ensure they are set in your .env file and VSCode terminal."
        )

    try:
        # Create authenticated client
        KITE_CLIENT = KiteApp(api_key=api_key)
        KITE_CLIENT.set_access_token(access_token)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Kite client: {e}")

    # Load instrument master immediately
    _load_instrument_map()


def _load_instrument_map():
    """
    Downloads and caches the instrument master table.

    Returns a DataFrame with columns such as:
        - instrument_token
        - exchange_token
        - tradingsymbol
        - name
        - last_price
        - expiry
        - etc.

    This table is *large* (~10k rows), but downloading it once at startup
    dramatically reduces response time for historical queries.
    """
    global KITE_CLIENT, INSTRUMENT_MAP

    if KITE_CLIENT is None:
        raise RuntimeError("Kite client not initialized.")

    try:
        instruments = KITE_CLIENT.instruments("NSE")
        if not isinstance(instruments, list):
            raise ValueError("Unexpected response from instruments(): expected list[dict]")

        df = pd.DataFrame(instruments)

        if "tradingsymbol" not in df.columns:
            raise ValueError("'tradingsymbol' missing from instrument master")

        # Fast lookup: symbol → row
        df.set_index("tradingsymbol", inplace=True)
        INSTRUMENT_MAP = df

        print("Kite Instrument Master loaded successfully.")

    except Exception as e:
        INSTRUMENT_MAP = pd.DataFrame()
        raise RuntimeError(f"Could not load instrument master: {e}")


# Run initialization ONCE when module is imported
_initialize_kite_client()


# =====================================================================================
# PUBLIC API: fetch_historical_data()
# =====================================================================================

def fetch_historical_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches historical OHLCV for a single symbol from Kite Connect.

    This function is used by:
        • FastAPI endpoint (/features/{symbol})
        • dataset generation pipeline
        • supervised GRU pretraining
        • RL environment bootstrapping

    Steps:
        1. Validate client and instrument map
        2. Resolve symbol → instrument_token
        3. Query kite.historical_data()
        4. Normalize into a clean pandas DataFrame with:
              Date (index)
              Open
              High
              Low
              Close
              Volume

    Returns:
        pd.DataFrame (empty if failure)
    """

    global KITE_CLIENT, INSTRUMENT_MAP

    # Ensure initialization did not fail
    if KITE_CLIENT is None:
        print("ERROR: Kite client not initialized.")
        return pd.DataFrame()

    if INSTRUMENT_MAP is None or INSTRUMENT_MAP.empty:
        print("ERROR: Instrument map not loaded.")
        return pd.DataFrame()

    # ----------------------------------------------------------------------
    # 1. Resolve symbol to instrument_token
    # ----------------------------------------------------------------------
    try:
        token = INSTRUMENT_MAP.loc[symbol.upper(), "instrument_token"]
    except KeyError:
        print(f"ERROR: Symbol '{symbol}' does not exist in instrument master.")
        return pd.DataFrame()

    # ----------------------------------------------------------------------
    # 2. Convert date strings
    # ----------------------------------------------------------------------
    try:
        from_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Use the provided end_date, or default to today if None
        if end_date:
            to_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            to_dt = datetime.now().date()
            
    except Exception as e:
        print(f"Date parsing error: {e}")
        return pd.DataFrame()

    # ----------------------------------------------------------------------
    # 3. Call historical_data()
    # ----------------------------------------------------------------------
    try:
        raw = KITE_CLIENT.historical_data(
            token,
            from_dt,
            to_dt,
            "day",
            continuous=False,
            oi=False
        )

        if not raw:
            print(f"No historical data returned for token {token}.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Kite API error while fetching historical data: {e}")
        return pd.DataFrame()

    # ----------------------------------------------------------------------
    # 4. Normalize schema → clean pandas DataFrame
    # ----------------------------------------------------------------------
    df = pd.DataFrame(raw)

    # Handle legacy field names returned by Kite versions
    if "date" in df.columns:
        df.rename(columns={"date": "Date"}, inplace=True)
    elif "timestamp" in df.columns:
        df.rename(columns={"timestamp": "Date"}, inplace=True)
    else:
        print(f"Unexpected OHLCV schema: {df.columns.tolist()}")
        return pd.DataFrame()

    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df
