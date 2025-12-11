from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <<< configure your FastAPI backend to explicitly tell browser it's safe for requests coming from frontend (http://localhost:3000).
import pandas as pd
import numpy as np
import requests # Used for making external API calls
from datetime import datetime, timedelta
# REMOVED: import pandas_ta as ta # We now use only stable Pandas/NumPy methods

# --- INITIALIZATION ---
app = FastAPI(title="Trading Agent Backend")

# --- CORS MIDDLEWARE FIX ---
# This is required to allow the frontend (running on port 3000) 
# to make API calls to the backend (running on port 8000).

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # Allow requests from your Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],           # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],
)
# ---------------------------

# ----------------------------------------------------------------------
## DATA INGESTION (REPLACE THIS FUNCTION)
# ----------------------------------------------------------------------

def fetch_historical_data_placeholder(symbol: str, start_date: str) -> pd.DataFrame:
    """
    Simulates fetching historical stock data using the 'requests' library.
    
    *** IMPORTANT ***
    When you integrate a real data API (e.g., Alpha Vantage, Polygon), 
    you MUST replace the data generation section (starting at 'Local Data Generation') 
    with your actual 'requests.get(...)' API call and JSON parsing logic.
    """
    
    # 1. Simulating the API Request Structure: (Template remains the same)
    # ...
    
    
    # 2. Local Data Generation (Temporary placeholder for testing):
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    except ValueError:
        # Fallback if start_date format is wrong
        start_dt = datetime.now() - timedelta(days=500)
    
    # Create a date range for simulation (Business Days)
    dates = pd.date_range(start=start_dt, end=datetime.now(), freq='B') 
    num_days = len(dates)
    
    if num_days < 50: # Check for minimum data required for features
        return pd.DataFrame()
        
    # Generate simulated price data (creating a realistic-looking time series)
    np.random.seed(42) 
    base_price = 100 + np.sin(np.arange(num_days) / 20) * 10 + np.random.randn(num_days) * 0.5
    
    data = {
        'Open': base_price * (1 + np.random.uniform(-0.01, 0.01, num_days)),
        'High': base_price * (1 + np.random.uniform(0.01, 0.02, num_days)),
        'Low': base_price * (1 + np.random.uniform(-0.02, -0.01, num_days)),
        'Close': base_price * (1 + np.random.uniform(-0.01, 0.01, num_days)),
        'Volume': np.random.randint(100000, 500000, num_days)
    }
    
    df = pd.DataFrame(data, index=dates) # generates a table with dates as indices, and other cols are Open, High, Low, Close, Volume (simple raw input features)
    df.index.name = 'Date'

    return df

# ----------------------------------------------------------------------
## FEATURE ENGINEERING (PURE PANDAS/NUMPY STABLE IMPLEMENTATION)
# ----------------------------------------------------------------------
    
def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates essential technical indicators (RSI, MACD, ATR) using only pure pandas/numpy.
    This guarantees stability by avoiding external library wrappers.
    """
    
    if len(df) < 30: 
        return pd.DataFrame() 
    
    # --- 1. Average True Range (ATR, timeperiod=14) ---
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'Prev_Close']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Prev_Close']), abs(row['Low'] - row['Prev_Close'])),
        axis=1
    )
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()

    # --- 2. Relative Strength Index (RSI, timeperiod=14) ---
    delta = df['Close'].diff()
    df['Avg_Gain'] = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    df['Avg_Loss'] = (-delta).where(delta < 0, 0).ewm(span=14, adjust=False).mean()
    
    RS = df['Avg_Gain'] / df['Avg_Loss']
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # --- 3. Moving Average Convergence Divergence (MACD, fast=12, slow=26, signal=9) ---
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- 4. FINAL CLEANUP ---
    
    # Select only the columns needed for the DRL agent and visualization
    feature_columns = ['Open', 'Close', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
    feature_df = df[feature_columns].copy()

    # Drop all rows that have NaN values (initial lookback periods)
    feature_df.dropna(inplace=True)
    
    # Final cleanup: Ensure all final columns are explicitly float
    if not feature_df.empty:
        for col in feature_columns:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(float)
    
    return feature_df

# ----------------------------------------------------------------------
## FASTAPI ENDPOINTS (These remain the same)
# ----------------------------------------------------------------------

@app.get("/") 
def read_root():
    """Simple health check endpoint. @app.get("/") registers a GET route on path / When someone visits GET http://host:port/ FastAPI calls read_root() and returns a JSON object. FastAPI automatically serializes the returned dict to JSON and sends Content-Type: application/json in the response header."""
    return {"message": "Trading Agent Backend is running!"}

@app.get("/api/v1/features/{symbol}")
def get_features(symbol: str, start_date: str = "2020-01-01"):
    """
    Endpoint that fetches raw data, calculates features, and returns the cleaned data. Symbol is given by URL, start_date is an optional query parameter with default value"2020-01-01".
    """
    # 1. Fetch raw data using the requests-based function
    raw_df = fetch_historical_data_placeholder(symbol, start_date)
    
    if raw_df.empty:
        raise HTTPException(status_code=404, detail=f"Could not fetch or generate data for {symbol}. Check your API connection.")
        
    # 2. Calculate features
    feature_df = calculate_technical_features(raw_df)
    
    if feature_df.empty:
        raise HTTPException(status_code=400, detail="Feature calculation failed (insufficient data remaining after calculating lookback windows).")

    # 3. Prepare for output
    feature_df.reset_index(inplace=True)
    # Convert DateTimeIndex back to string for clean JSON serialization
    feature_df['Date'] = feature_df['Date'].dt.strftime('%Y-%m-%d') 
    
    # Define (= whitelist) the final columns for the RL agent and visualization
    feature_columns = ['Date', 'Open', 'Close', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
    
    # Return the last 100 days of windowed features (Week 1 goal)
    return {
        "symbol": symbol,
        "features": feature_df[feature_columns].tail(100).to_dict('records')
    }
    # features is a list of objects where each object contains Date as a string (YYYY-MM-DD) and numeric fields. The CharingDashboard.tsx component’s FeatureData TypeScript interface matches that exactly!