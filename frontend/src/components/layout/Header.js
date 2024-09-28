import React from 'react';
import './Header.css'; // Optional CSS file for styling

const Header = () => {
  return (
    <div className="header">
      <h1 className="header-title">Stock Price Prediction Dashboard</h1>
      <p className="header-subtitle">
        Use various machine learning models to predict stock prices.
      </p>
    </div>
  );
};

export default Header;
