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
  // Placeholder for the trading action, which will be filled by the RL agent later
  Action: 'Buy' | 'Sell' | 'Hold'; 
}

// Custom Tooltip component for better data display on hover
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload;
    const action = dataPoint.Action || 'N/A';
    
    // Determine a color based on a simulated action (will be dynamic later)
    const actionColor = action === 'Buy' 
      ? '#2563EB' // Blue (for Buy/Positive)
      : action === 'Sell'
      ? '#F97316' // Orange (for Sell/Negative)
      : '#A1A1AA'; // Gray (for Hold/Neutral)

    return (
      <div className="p-2 border border-gray-300 bg-white shadow-lg rounded">
        <p className="font-bold text-sm">Date: {label}</p>
        <p className="text-sm">Close: ${dataPoint.Close.toFixed(2)}</p>
        <p className="text-sm">RSI: {dataPoint.RSI.toFixed(2)}</p>
        <p className="text-sm">MACD: {dataPoint.MACD.toFixed(2)}</p>
        <p className="text-sm" style={{ color: actionColor }}>Action: {action}</p>
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
  
  // Default parameters for the API call
  const symbol = "AAPL"; 
  const startDate = "2023-01-01";

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch features from the FastAPI endpoint (for now, I use the API-endpoint exposed by my own backend running on port 8000; later to be replaced with external API endpoint!)
        const response = await axios.get(`http://localhost:8000/api/v1/features/${symbol}`, {
          params: { start_date: startDate }
        });

        // Map the fetched data and inject a temporary 'Action' for visualization; takes feature-array returned by FastAPI backend and maps it: item = 1 row, index = index of that row (index 0 is oldest, later numbers newer datapoints) coming from feature-arry of backend) 
        const processedData: FeatureData[] = response.data.features.map((item: any, index: number) => ({ 
          ...item,
          // Temporary logic: Buy on the 10th and 50th data points, Sell on the 30th and 70th
          // Artificially injects some Buy/Sell markers so that the Recharts chart can display something interesting.
          Action: (index === 10 || index === 50) ? 'Buy' : 
                  (index === 30 || index === 70) ? 'Sell' : 'Hold'
        }));
        
        setData(processedData);
        
      } catch (err) {
        console.error("Error fetching data:", err);
        setError("Failed to load data from backend API. Please check your terminal and ensure the container is running and accessible on port 8000.");
        setData([]); 
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []); // Empty dependency array: run once on mount

  if (loading) {
    return <div className="text-center p-10">Loading feature data...</div>;
  }

  if (error) {
    return <div className="text-center p-10 text-xl font-bold text-red-600">{error}</div>;
  }
  
  return (
    <div className="p-6 bg-white shadow-xl rounded-lg">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">
        Trading Simulation Viewer: {symbol} (Using Features API)
      </h2>
      
      {/* Chart controls placeholder */}
      <div className="flex space-x-4 mb-6">
        <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Run Simulation</button>
        <span className="p-2 text-gray-500">Dates and Window Length TBD</span>
      </div>

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