import React from 'react';
import './Input.css'; // Optional CSS file for styling

const Input = ({ label, type, value, onChange }) => {
  return (
    <div className="input-container">
      {label && <label className="input-label">{label}</label>}
      <input
        type={type}
        value={value}
        onChange={onChange}
        className="input-field"
      />
    </div>
  );
};

export default Input;
