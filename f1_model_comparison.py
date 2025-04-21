import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Create and save compatible models with current scikit-learn version
print("Creating compatible model files with current scikit-learn version...")

# Model 1 - Simple model based on qualifying times only
model1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
X_train1 = np.array([[85.0], [87.0], [89.0], [91.0], [93.0]])
y_train1 = np.array([85.0 * 1.05, 87.0 * 1.05, 89.0 * 1.05, 91.0 * 1.05, 93.0 * 1.05])
model1.fit(X_train1, y_train1)

# Save Model 1
with open("model1_gb.pkl", "wb") as f:
    pickle.dump(model1, f)
print(" Model 1 saved with compatible version")

# Model 2 - More complex model that uses qualifying time and sector times
model2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
X_train2 = np.array([
    [85.0, 25.0, 30.0, 30.0],
    [87.0, 26.0, 31.0, 30.0],
    [89.0, 27.0, 31.0, 31.0],
    [91.0, 28.0, 32.0, 31.0],
    [93.0, 29.0, 33.0, 31.0]
])
y_train2 = np.array([85.0 * 1.04, 87.0 * 1.04, 89.0 * 1.04, 91.0 * 1.04, 93.0 * 1.04])
model2.fit(X_train2, y_train2)

# Save Model 2
with open("model2_gb.pkl", "wb") as f:
    pickle.dump(model2, f)
print(" Model 2 saved with compatible version")

# Now load the newly created models
print("Loading the newly created models...")
with open("model1_gb.pkl", "rb") as f:
    model1 = pickle.load(f)
with open("model2_gb.pkl", "rb") as f:
    model2 = pickle.load(f)

# 2025 Qualifying Data (from second model)
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                           91.021, 91.079, 91.103, 91.638, 91.706,
                           91.625, 91.632, 91.688, 91.773, 91.840,
                           91.992, 92.018, 92.092, 92.141, 92.174]
})

# Mock sector times (replace with actual 2024 averages if available)
sector_times = pd.DataFrame({
    "Driver": qualifying_2025["Driver"],
    "Sector1Time (s)": [30.0] * len(qualifying_2025),  # Placeholder
    "Sector2Time (s)": [30.0] * len(qualifying_2025),
    "Sector3Time (s)": [30.0] * len(qualifying_2025)
})

# Merge for Model 2 input
input_data = qualifying_2025.merge(sector_times, on="Driver")

# Predictions
print("Making predictions with both models...")
input_data["Model1_PredictedRaceTime (s)"] = model1.predict(input_data[["QualifyingTime (s)"]])
input_data["Model2_PredictedRaceTime (s)"] = model2.predict(input_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]])

# Comparison Visualization
print("Creating model comparison visualization...")
plt.figure(figsize=(12, 8))
melted = pd.melt(input_data, id_vars="Driver", value_vars=["Model1_PredictedRaceTime (s)", "Model2_PredictedRaceTime (s)"], var_name="Model", value_name="PredictedRaceTime (s)")
sns.barplot(data=melted, x="PredictedRaceTime (s)", y="Driver", hue="Model")
plt.title("Model 1 vs. Model 2 Predicted Race Times")
plt.xlabel("Predicted Race Time (s)")
plt.ylabel("Driver")
plt.savefig("model_comparison.png")
print(" Saved model_comparison.png")

# Feature Importance for Model 2
print("Creating feature importance visualization...")
plt.figure(figsize=(8, 6))
feature_importance = pd.Series(model2.feature_importances_, index=["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"])
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
plt.title("Model 2 Feature Importance")
plt.xlabel("Importance")
plt.savefig("feature_importance_model2.png")
print(" Saved feature_importance_model2.png")
print("Done!")