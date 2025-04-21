# F1 2025 Chinese GP Race Time Prediction

## Overview

This project predicts Formula 1 race lap times for the 2025 Chinese GP using two Gradient Boosting models:

- **Model 1**: Uses 2024 Australian GP lap times and 2025 qualifying times (single feature).
- **Model 2**: Uses 2024 Chinese GP lap and sector times, plus 2025 qualifying times (multiple features). The project includes exploratory data analysis (EDA), model training, and a Streamlit web app for interactive predictions.

## Features

- **Data**: FastF1 2024 Australian and Chinese GP data, custom 2025 Chinese GP qualifying times.
- **EDA**: Visualizations of lap times, sector times, qualifying times, and correlations.
- **Models**: Two Gradient Boosting Regressors with different feature sets.
- **Web App**: Streamlit app to input qualifying/sector times and view predictions.
- **Metrics**: Mean Absolute Error (MAE) for model evaluation.

## Installation

```bash
pip install fastf1 pandas scikit-learn matplotlib seaborn streamlit
```

## Usage

1. **EDA**: Run `f1_eda_2025.py` to generate visualizations (`f1_eda_2025_visualizations.png`).
2. **Train Models**: Run the provided scripts and save models as `model1_gb.pkl` and `model2_gb.pkl`.
3. **Run Web App**:

   ```bash
   streamlit run f1_prediction_app_2025.py
   ```
4. **Compare Models**: Run `f1_model_comparison.py` to visualize model differences.
5. **Deploy**: Push to GitHub and deploy on Streamlit Cloud.

## Results

- **Model 1 MAE**: \[Replace with your MAE\].
- **Model 2 MAE**: \[Replace with your MAE\].
- **Top Predicted Drivers**: Oscar Piastri, George Russell, Lando Norris (based on Model 2).
- **Visualizations**: See `f1_eda_2025_visualizations.png`, `model_comparison.png`, `feature_importance_model2.png`.

## Data Sources

- **FastF1**: 2024 Australian and Chinese GP lap and sector times.
- **Custom Data**: 2025 Chinese GP qualifying times (simulated).

## Challenges

- Missing sector times for new drivers in Model 2.
- Model 1 uses Australian GP data, which may not generalize to Chinese GP.
- Limited features (e.g., no tyre or weather data).

## Future Improvements

- Impute missing sector times using track averages or predictive models.
- Add features like tyre compounds, weather, or pit stop strategies.
- Extend to other 2025 races.
- Experiment with other models (e.g., XGBoost, Neural Networks).

## Links

- Streamlit App
- Medium Article
- Slide Deck

## License

MIT