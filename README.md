# 🚍 Bus Travel Time Prediction (Zero-Shot)

## Overview
This project predicts bus travel times without using GPS trajectory data.  
Instead, it uses static trip attributes and road characteristics to generalise to unseen routes.

## Approach
- Built a feature-rich dataset from:
  - Road segments (e.g. length, speed limits)
  - Trip-level attributes
- Compared four models:
  - Linear Regression  
  - XGBoost  
  - MLP  
  - LSTM (sequential road features)
- Implemented a full ML pipeline: preprocessing, training, evaluation

## Results
- Neural networks outperformed traditional models  
- No significant difference between MLP and LSTM  
- Distance and max speed are the most important predictors  

## Key Insight
Accurate travel time prediction is feasible without historical data, enabling:
- Predictions for new routes  
- Use in data-scarce environments  
- Support for transport planning & scheduling  

## Tech Stack
Python · PyTorch/TensorFlow · Scikit-learn · XGBoost · Pandas · QGIS · OpenStreetMap

## Context
MSc Artificial Intelligence thesis, conducted in collaboration with Irias Informatiemanagement.
