// src/components/charts/StockPriceChart.js
import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import Papa from 'papaparse';

// Import necessary chart.js components and register them
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register the necessary components globally
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function StockPriceChart({ selectedStock, startDate, endDate, graphOptions }) {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    // Load the CSV file using papaparse
    Papa.parse("/data/processed/processed_data_with_predictions.csv", {
      download: true,
      header: true,
      complete: (result) => {
        const data = result.data;

        // Filter data based on the selected stock, date range, and graph options
        const filteredData = data.filter(
          (row) =>
            row.Name === selectedStock &&
            new Date(row.date) >= new Date(startDate) &&
            new Date(row.date) <= new Date(endDate)
        );

        // Prepare data for the chart
        const labels = filteredData.map((row) => row.date);
        const datasets = [];

        if (graphOptions.includes("actual")) {
          datasets.push({
            label: "Actual Values",
            data: filteredData.map((row) => parseFloat(row.close)),
            borderColor: 'rgba(75,192,192,1)',
            fill: false,
          });
        }
        if (graphOptions.includes("svr_predicted")) {
          datasets.push({
            label: "SVR Predicted",
            data: filteredData.map((row) => parseFloat(row.svr_predicted)),
            borderColor: 'rgba(153, 102, 255, 1)',
            fill: false,
          });
        }
        if (graphOptions.includes("rf_predicted")) {
          datasets.push({
            label: "RF Predicted",
            data: filteredData.map((row) => parseFloat(row.rf_predicted)),
            borderColor: 'rgba(255, 159, 64, 1)',
            fill: false,
          });
        }

        // Set the chart data
        setChartData({
          labels,
          datasets,
        });
      },
    });
  }, [selectedStock, startDate, endDate, graphOptions]);

  return (
    <div>
      {chartData ? (
        <Line
          data={chartData}
          options={{
            title: {
              display: true,
              text: `Stock Price Predictions for ${selectedStock}`,
              fontSize: 20,
            },
            scales: {
              x: { title: { display: true, text: "Date" } },
              y: { title: { display: true, text: "Stock Price" } },
            },
          }}
        />
      ) : (
        <p>Loading chart data...</p>
      )}
    </div>
  );
}

export default StockPriceChart;
