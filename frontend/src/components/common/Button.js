import React from 'react';
import './Button.css'; // Optional CSS file for styling

const Button = ({ text, onClick, style }) => {
  return (
    <button className="common-button" style={style} onClick={onClick}>
      {text}
    </button>
  );
};

export default Button;
