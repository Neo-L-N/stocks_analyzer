// src/App.js

import React, { useState } from 'react';
import DateRangePicker from './components/common/DateRangePicker'; // Correct path
import StockPriceChart from './components/charts/StockPriceChart';


function App() {
  const [startDate, setStartDate] = useState(new Date("2013-01-01")); // Set your start date
  const [endDate, setEndDate] = useState(new Date("2018-01-01")); // Set your end date

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Prediction Dashboard</h1>
        <DateRangePicker
          startDate={startDate}
          endDate={endDate}
          onStartDateChange={setStartDate}
          onEndDateChange={setEndDate}
        />
        <StockPriceChart
          startDate={startDate}
          endDate={endDate}
          selectedStock="AAPL" // Example stock, use state if needed
          graphOptions={["actual", "svr_predicted", "rf_predicted"]}
        />
      </header>
    </div>
  );
}

export default App;

