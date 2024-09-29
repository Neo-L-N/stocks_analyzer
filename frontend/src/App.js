// src/App.js

import React, { useState } from 'react';
import DateRangePicker from './components/common/DateRangePicker';
import StockPriceChart from './components/charts/StockPriceChart';
import ChatBot from './components/chatbot/ChatBot'; // Import the ChatBot component

function App() {
  const [startDate, setStartDate] = useState(new Date("2013-01-01"));
  const [endDate, setEndDate] = useState(new Date("2018-01-01"));

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
          selectedStock="AAPL"
          graphOptions={["actual", "svr_predicted", "rf_predicted"]}
        />
      </header>
      {/* Add the ChatBot component */}
      <ChatBot />
    </div>
  );
}

export default App;
