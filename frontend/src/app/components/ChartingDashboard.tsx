//This component will render the candlestick chart using dummy data for now.

"use client"; // run in Browser, not Server or backend

import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import axios from 'axios';

// Define the interface for the data we expect from the backend /features endpoint
interface FeatureData {
  Date: string;
  Open: number;
  Close: number;
  RSI: number;
  MACD: number;
  MACD_Signal: number;
  ATR: number;
  SMA_50: number;    
  OBV: number;       
  ROC_10: number;    
  SMA_Ratio: number; 
  RealizedVol_20: number;
  Action: 'Buy' | 'Sell' | 'Hold';   // Placeholder for the trading action, which will be filled by the RL agent later
}


// Custom Tooltip component for better data display on hover
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload as FeatureData; // Use FeatureData type
    const action = dataPoint.Action || 'N/A';
    
    const actionColor = action === 'Buy' 
      ? '#2563EB' // Blue
      : action === 'Sell'
      ? '#F97316' // Orange
      : '#A1A1AA'; // Gray

    return (
      <div className="p-2 border border-gray-300 bg-white shadow-lg rounded">
        <p className="font-bold text-sm">Date: {label}</p>
        <hr className="my-1"/>
        <p className="text-sm">Close: ${dataPoint.Close.toFixed(2)}</p>
        <p className="text-sm" style={{ color: actionColor }}>Action: {action}</p>
        <hr className="my-1"/>
        <p className="text-sm">RSI: {dataPoint.RSI.toFixed(2)}</p>
        <p className="text-sm">MACD: {dataPoint.MACD.toFixed(2)} / Signal: {dataPoint.MACD_Signal.toFixed(2)}</p>
        <p className="text-sm">ATR: {dataPoint.ATR.toFixed(2)}</p>
        <p className="text-sm">SMA_50: {dataPoint.SMA_50.toFixed(2)}</p>
        <p className="text-sm">OBV: {dataPoint.OBV.toFixed(0)}</p>
        <p className="text-sm">ROC_10: {dataPoint.ROC_10.toFixed(2)}%</p>
        <p className="text-sm">SMA_Ratio: {dataPoint.SMA_Ratio.toFixed(4)}</p>
        <p className="text-sm">RealizedVol_20: {dataPoint.RealizedVol_20.toFixed(4)}</p>
      </div>
    );
  }
  return null;
};

//React components don’t store data in variables like Python.
//They store data in state, because React automatically re-renders the UI when state changes:

const ChartingDashboard: React.FC = () => {
  // We use the FeatureData interface for the state
  const [data, setData] = useState<FeatureData[]>([]); // Initializes state. <FeatureData[]> tells TypeScript what type the state holds: → initially empty array of FeatureData objects.
  // You start with data = [], whereas setData is a function used to update the data
  // You fetch data from backend asynchronously
  // When the backend returns, you call: setData(processedData) -> React sees it and re-renders the component accordingly!

  const [loading, setLoading] = useState(true); // will show loading screen as long as data is being fetched by backend; then (see useEffect), after successful fetch: setLoading(false)

  const [error, setError] = useState<string | null>(null);
  
  // --- 1. NEW STATE: Symbol Input and Active Symbol ---
  // This holds the symbol currently displayed/fetched
  const [activeSymbol, setActiveSymbol] = useState("TCS"); 
  // This holds the symbol the user types into the input box
  const [inputSymbol, setInputSymbol] = useState("TCS");
  
  // Default parameters (start date is fixed for simplicity)
  const startDate = "2021-01-01"; 

  // --- 2. NEW FUNCTION: Handle Fetch Trigger ---
  const handleFetch = () => {
    // Only fetch if the input symbol is different or if it's the initial fetch
    if (inputSymbol.trim().toUpperCase() !== activeSymbol) {
      setActiveSymbol(inputSymbol.trim().toUpperCase());
    }
  };
  
  // --- 3. REVISED useEffect: Triggered by activeSymbol change ---
  useEffect(() => {
    // Only fetch if activeSymbol is set (i.e., after the first render)
    if (!activeSymbol) return; 
    
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // The symbol in the URL is now the state variable, not a constant
        const response = await axios.get(`http://localhost:8000/api/v1/features/${activeSymbol}`, {
          params: { start_date: startDate }
        });

        // ... (rest of the processing logic remains the same) ...
        const processedData: FeatureData[] = response.data.features.map((item: any, index: number) => ({ 
          ...item,
          Action: (index === 10 || index === 50) ? 'Buy' : 
                  (index === 30 || index === 70) ? 'Sell' : 'Hold'
        }));
        
        setData(processedData);
        
      } catch (err) {
        console.error("Error fetching data:", err);
        setError(`Failed to load data for ${activeSymbol}. Check backend logs or symbol spelling.`);
        setData([]); 
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [activeSymbol]); // <-- CRITICAL: Dependency array now includes activeSymbol

  if (loading) {
    return <div className="text-center p-10">Loading feature data...</div>;
  }

  if (error) {
    return <div className="text-center p-10 text-xl font-bold text-red-600">{error}</div>;
  }
  
  return (
    <div className="p-6 bg-white shadow-xl rounded-lg">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">
        Trading Simulation Viewer: {activeSymbol} (Using Features API)
      </h2>
      
      {/* --- 4. NEW: Input and Button for Dynamic Symbol Selection --- */}
      <div className="flex space-x-4 mb-6 items-center">
        <input
          type="text"
          value={inputSymbol}
          onChange={(e) => setInputSymbol(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleFetch(); }}
          className="px-3 py-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
          placeholder="Enter Ticker (e.g., TCS)"
        />
        <button 
          onClick={handleFetch}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition duration-150"
        >
          Fetch Symbol
        </button>
        {/* Placeholder for the Run Simulation button */}
        <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Run Simulation</button>
      </div>
      {/* ------------------------------------------------------------------ */}

      <div style={{ width: '100%', height: 400 }}>
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              {/* XAxis using the 'Date' field from your features data */}
              <XAxis dataKey="Date" stroke="#333" interval="preserveStartEnd" tickFormatter={(date) => date.slice(5)} /> 
              {/* YAxis for the price */}
              <YAxis domain={['dataMin', 'dataMax']} stroke="#333" orientation="left" /> 
              <Tooltip content={<CustomTooltip />} />
              
              {/* Plot the Close price */}
              <Line 
                type="monotone" 
                dataKey="Close" 
                stroke="#2563EB" // Blue for price line
                strokeWidth={2}
                dot={({ cx, cy, payload }) => {
                  // Custom dot logic to display Buy/Sell markers
                  const action = payload.Action;
                  let fill = '#999999'; // Default Gray for Hold
                  let r = 0; // Default radius

                  if (action === 'Buy') {
                    fill = '#2563EB'; // Blue
                    r = 5;
                  } else if (action === 'Sell') {
                    fill = '#F97316'; // Orange
                    r = 5;
                  }
                  
                  // Only draw a visible dot for Buy/Sell actions
                  return <circle cx={cx} cy={cy} r={r} fill={fill} stroke="#FFF" strokeWidth={1} />;
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center p-20 text-gray-500">No trading data available to display.</div>
        )}
      </div>
      <p className="mt-4 text-sm text-gray-500">
        *Blue markers indicate a simulated 'Buy' action, and Orange markers indicate a simulated 'Sell' action.
      </p>
    </div>
  );
};

export default ChartingDashboard;

// Program execution flow:
// 1. When your component mounts (first render):

// The function ChartingDashboard() runs once.

// It initializes all the state (data, loading, error).

// It hits the useEffect(..., []) hook — which schedules fetchData() to run after render.

// It reaches this part:

// if (loading) {
//   return <div>Loading feature data...</div>;
// }
// Since loading is initially true, your component returns the loading screen JSX.

// Rendering stops here.

// 2. React now executes: fetchData();

// Inside fetchData:

// You call your backend API

// Process the results

// Run setData(processedData)

// Run setLoading(false)

// These state updates cause React to re-run the entire component function.

// This is the key mental model shift:

// 🎯 Every time state updates → your component function runs again.

// 3. Second render:

// Now state is:

// loading = false

// error = null

// data = [ …100 feature rows… ]

// So React runs your component again and now evaluates:

// if (loading) { ... } // no, loading = false
// if (error) { ... }   // no

// Then it reaches your “main” JSX:

// return (
//   <div>
//     <ResponsiveContainer>
//       <LineChart data={data}>

// --> Chart appears!</LineChart>