import React from 'react';
import './StockWidget.css'; // Optional CSS file for styling

const StockWidget = ({ name, price, change }) => {
  return (
    <div className={`stock-widget ${change >= 0 ? 'up' : 'down'}`}>
      <h3>{name}</h3>
      <p>Price: ${price.toFixed(2)}</p>
      <p className={`stock-change ${change >= 0 ? 'positive' : 'negative'}`}>
        {change >= 0 ? `+${change.toFixed(2)}` : change.toFixed(2)}%
      </p>
    </div>
  );
};

export default StockWidget;
