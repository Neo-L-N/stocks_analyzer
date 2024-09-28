// src/App.js

import React, { useState } from 'react';
import './App.css';
import StockPriceChart from './components/charts/StockPriceChart';

function App() {
  // State management for the selected stock, date range, and graph options
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [startDate, setStartDate] = useState("2015-01-01");
  const [endDate, setEndDate] = useState("2018-12-31");
  const [graphOptions, setGraphOptions] = useState(["actual", "svr_predicted", "rf_predicted"]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Prediction Dashboard</h1>
        <p>Select a stock, date range, and view predictions:</p>
      </header>

      {/* Render the stock price chart with appropriate props */}
      <StockPriceChart
        selectedStock={selectedStock}
        startDate={startDate}
        endDate={endDate}
        graphOptions={graphOptions}
      />
    </div>
  );
}

export default App;

