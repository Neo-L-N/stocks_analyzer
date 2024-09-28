// ChartComponent.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

const ChartComponent = ({ stock, startDate, endDate }) => {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    axios.get(`http://127.0.0.1:5000/api/chart-data`, {
      params: { stock, start_date: startDate, end_date: endDate }
    })
    .then(response => {
      const { date, actual, svr_predicted, rf_predicted } = response.data;
      setChartData({
        labels: date,
        datasets: [
          {
            label: 'Actual Values',
            data: actual,
            borderColor: 'rgba(75, 192, 192, 1)',
            fill: false,
          },
          {
            label: 'SVR Predicted',
            data: svr_predicted,
            borderColor: 'rgba(255, 99, 132, 1)',
            fill: false,
          },
          {
            label: 'RF Predicted',
            data: rf_predicted,
            borderColor: 'rgba(54, 162, 235, 1)',
            fill: false,
          }
        ]
      });
    })
    .catch(error => console.error('Error fetching chart data:', error));
  }, [stock, startDate, endDate]);

  return chartData ? (
    <Line
      data={chartData}
      options={{
        responsive: true,
        title: { display: true, text: `Stock Price Predictions for ${stock}` },
        scales: {
          xAxes: [{ type: 'time', time: { unit: 'day' } }]
        }
      }}
    />
  ) : (
    <p>Loading chart data...</p>
  );
};

export default ChartComponent;
