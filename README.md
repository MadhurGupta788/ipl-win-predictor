# IPL Win Probability Predictor 🏏

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-XGBoost-green)
![Web App](https://img.shields.io/badge/Streamlit-Deployed-red)

## Overview
A machine learning web application that predicts the real-time, ball-by-ball win probability for the chasing team in an Indian Premier League (IPL) match. 

Unlike standard binary classification models that simply output a final Win/Loss prediction, this project heavily utilizes **probability calibration** to output realistic percentages that react dynamically to the match situation—capturing the non-linear pressure of falling wickets and climbing run rates.

## 🚀 Live Application
**[Play with the Live Predictor Here](https://ipl-win-predictor-788.streamlit.app/)

## 🧠 Architecture & Engineering

### Feature Engineering
The model derives dynamic match context from raw Kaggle ball-by-ball data. To give the algorithm an understanding of the game state, the following features are calculated for every single delivery:
* **Pace Metrics:** Current Run Rate (`crr`) and Required Run Rate (`rrr`)
* **Game State:** `runs_left`, `balls_left`, `wickets_left`
* **Match Context:** Batting team, bowling team, and host city.

### Model Pipeline
1. **Baseline:** Established a robust baseline using `LogisticRegression` within a Scikit-Learn `Pipeline` with One-Hot Encoding for categorical variables.
2. **Advanced Model:** Upgraded to `XGBoost` to capture complex, non-linear game dynamics (e.g., the compounding impact of losing 3 wickets in the powerplay versus the death overs).
3. **Calibration:** Wrapped the XGBoost classifier in Scikit-Learn's `CalibratedClassifierCV`. Because tree-based models tend to push predictions toward extreme 0s or 1s, calibration ensures the model outputs reliable, real-world probability percentages.

## 🛠️ Tech Stack
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Web Framework:** `streamlit`
* **Deployment:** Streamlit Community Cloud

## 💻 Local Setup & Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/ipl-win-predictor.git](https://github.com/your-username/ipl-win-predictor.git)
   cd ipl-win-predictor
