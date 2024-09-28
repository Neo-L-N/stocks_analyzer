#!/bin/bash

echo "Navigating to backend directory..."
# Update this path to reflect the correct absolute path to the backend directory
cd ~/Documents/Projects/stocks_analysis_alpha/backend/code || { echo "Backend directory not found"; exit 1; }

echo "Starting backend server..."
# Run backend server in the background and redirect logs to backend.log
nohup python app.py > ../../backend.log 2>&1 &

# Wait for a few seconds to ensure the backend starts properly
sleep 5

echo "Navigating to frontend directory..."
# Update this path to reflect the correct absolute path to the frontend directory
cd ~/Documents/Projects/stocks_analysis_alpha/frontend || { echo "Frontend directory not found"; exit 1; }

echo "Starting frontend server..."
# Run frontend server in the background and redirect logs to frontend.log
nohup npm start > ../frontend.log 2>&1 &

echo "Servers started successfully!"


