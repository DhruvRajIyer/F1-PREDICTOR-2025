import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import fastf1
import base64
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="F1 2025 Chinese GP Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Load or create models
model1 = None
model2 = None

try:
    if os.path.exists("model1_gb.pkl"):
        with open("model1_gb.pkl", "rb") as f:
            model1 = pickle.load(f)
    if os.path.exists("model2_gb.pkl"):
        with open("model2_gb.pkl", "rb") as f:
            model2 = pickle.load(f)
except Exception as e:
    st.warning(f"Error loading models: {e}")

# Create placeholder models if they don't exist
if model1 is None:
    model1 = GradientBoostingRegressor(n_estimators=10, random_state=39)
    X_train = np.array([[85.0], [87.0], [89.0], [91.0], [93.0]])
    y_train = np.array([85.0 * 1.05, 87.0 * 1.05, 89.0 * 1.05, 91.0 * 1.05, 93.0 * 1.05])
    model1.fit(X_train, y_train)
    with open("model1_gb.pkl", "wb") as f:
        pickle.dump(model1, f)

if model2 is None:
    model2 = GradientBoostingRegressor(n_estimators=10, random_state=38)
    X_train = np.array([
        [85.0, 25.0, 30.0, 30.0],
        [87.0, 26.0, 31.0, 30.0],
        [89.0, 27.0, 31.0, 31.0],
        [91.0, 28.0, 32.0, 31.0],
        [93.0, 29.0, 33.0, 31.0]
    ])
    y_train = np.array([85.0 * 1.04, 87.0 * 1.04, 89.0 * 1.04, 91.0 * 1.04, 93.0 * 1.04])
    model2.fit(X_train, y_train)
    with open("model2_gb.pkl", "wb") as f:
        pickle.dump(model2, f)

# Function to display code with syntax highlighting
def display_code(file_path):
    try:
        with open(file_path, 'r') as file:
            code = file.read()
            st.code(code, language='python')
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")

# Function to read and run a Python file with captured output
def run_code_with_output(file_path):
    import sys
    from io import StringIO
    import contextlib
    
    # Read the file
    with open(file_path, 'r') as file:
        code = file.read()
    
    # Create a StringIO object to capture output
    output = StringIO()
    
    # Redirect stdout to our StringIO object
    with contextlib.redirect_stdout(output):
        try:
            exec(code)
            result = output.getvalue()
            return result
        except Exception as e:
            return f"Error executing code: {str(e)}"

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧪 Models", "🔮 Predictor"])

# Tab 1: EDA (Exploratory Data Analysis)
with tab1:
    st.title("🏎️ F1 2025 Chinese GP Data Exploration")
    st.write("This tab shows exploratory data analysis of F1 race data.")
    
    # Load data for EDA
    try:
        st.subheader("Loading F1 Race Data")
        
        # Check if cache folder exists
        if not os.path.exists("f1_cache"):
            os.makedirs("f1_cache")
            
        # Enable FastF1 caching
        fastf1.Cache.enable_cache("f1_cache")
        
        with st.spinner("Loading 2024 Chinese GP race data..."):
            session_2024 = fastf1.get_session(2024, "China", "R")
            session_2024.load()
            
            # Extract lap and sector times
            laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
            laps_2024.dropna(inplace=True)
            
            for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()
            
            # Group by driver for average sector times
            sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
            
            # 2025 Qualifying Data
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
            
            # Driver mapping
            driver_mapping = {
                "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
                "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
                "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
                "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
                "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
            }
            qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)
            
            # Merge data
            merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")
            
            st.success("Data loaded successfully!")
        
        # Display data tables
        st.subheader("2024 Chinese GP Lap Times")
        st.dataframe(laps_2024.head(10))
        
        st.subheader("Average Sector Times by Driver")
        st.dataframe(sector_times_2024)
        
        st.subheader("2025 Qualifying Times")
        st.dataframe(qualifying_2025[["Driver", "QualifyingTime (s)"]])
        
        # Create visualizations
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Qualifying Time Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=qualifying_2025.sort_values(by="QualifyingTime (s)"), 
                        x="QualifyingTime (s)", y="Driver", palette="viridis", ax=ax)
            plt.title("2025 Chinese GP Qualifying Times")
            st.pyplot(fig)
        
        with col2:
            st.write("#### Sector Time Comparison")
            sector_times_melted = pd.melt(sector_times_2024, id_vars=["Driver"], 
                                        value_vars=["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"],
                                        var_name="Sector", value_name="Time (s)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=sector_times_melted, x="Sector", y="Time (s)", ax=ax)
            plt.title("Sector Time Distribution - 2024 Chinese GP")
            st.pyplot(fig)
        
        st.subheader("Correlation Analysis")
        sector_merged = merged_data.dropna()
        if not sector_merged.empty:
            corr_matrix = sector_merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            plt.title("Correlation Matrix between Qualifying and Sector Times")
            st.pyplot(fig)
        else:
            st.warning("Not enough data for correlation analysis")
            
    except Exception as e:
        st.error(f"Error in data analysis: {str(e)}")
        st.info("If FastF1 is not installed, you can install it with: pip install fastf1")

# Tab 2: Models
with tab2:
    st.title("🏎️ F1 2025 Chinese GP Prediction Models")
    st.write("This tab shows the prediction models and their interactive controls.")
    
    # Load data for analysis
    try:
        # Check if cache folder exists
        if not os.path.exists("f1_cache"):
            os.makedirs("f1_cache")
            
        # Enable FastF1 caching
        fastf1.Cache.enable_cache("f1_cache")
        
        with st.spinner("Loading F1 data..."):
            # Load 2024 Chinese GP race session
            session_2024 = fastf1.get_session(2024, "China", "R")
            session_2024.load()
            
            # Extract lap and sector times
            laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
            laps_2024.dropna(inplace=True)
            
            for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()
            
            # Group by driver for average sector times
            sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "LapTime (s)"]].mean().reset_index()
            
            # 2025 Qualifying Data
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
            
            # Driver mapping
            driver_mapping = {
                "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
                "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
                "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
                "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
                "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
            }
            
            # Reverse mapping (for display purposes)
            reverse_mapping = {v: k for k, v in driver_mapping.items()}
            
            qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)
            
            # Merge data
            merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", 
                                                how="left", suffixes=('_full', '_code'))
            
            st.header("Driver Analysis")
            st.write("Select a driver to see their data and predictions")
            
            # Create list of available drivers with data
            available_drivers = merged_data.dropna(subset=["Sector1Time (s)"])["Driver_full"].tolist()
            if not available_drivers:
                available_drivers = qualifying_2025["Driver"].tolist()
            
            selected_driver = st.selectbox("Select Driver", 
                                          options=available_drivers,
                                          index=min(3, len(available_drivers)-1))
            
            # Get driver data
            driver_data = merged_data[merged_data["Driver_full"] == selected_driver].iloc[0]
            
            # Display driver stats in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Qualifying Time", f"{driver_data['QualifyingTime (s)']:.3f}s")
                
            with col2:
                if pd.notna(driver_data.get("Sector1Time (s)")):
                    sector_total = driver_data["Sector1Time (s)"] + driver_data["Sector2Time (s)"] + driver_data["Sector3Time (s)"]
                    st.metric("Avg. Sector Total", f"{sector_total:.3f}s")
                else:
                    st.metric("Avg. Sector Total", "No Data")
                    
            with col3:
                if pd.notna(driver_data.get("LapTime (s)")):
                    st.metric("Avg. Race Lap (2024)", f"{driver_data['LapTime (s)']:.3f}s")
                else:
                    st.metric("Avg. Race Lap (2024)", "No Data")
            
            # Sector time breakdown
            if pd.notna(driver_data.get("Sector1Time (s)")):
                st.subheader("Sector Time Breakdown")
                sector_cols = st.columns(3)
                
                with sector_cols[0]:
                    st.metric("Sector 1", f"{driver_data['Sector1Time (s)']:.3f}s")
                    
                with sector_cols[1]:
                    st.metric("Sector 2", f"{driver_data['Sector2Time (s)']:.3f}s")
                    
                with sector_cols[2]:
                    st.metric("Sector 3", f"{driver_data['Sector3Time (s)']:.3f}s")
            
            # Prepare data for model predictions
            X1 = merged_data[["QualifyingTime (s)"]].fillna(0)
            
            # Filter out rows with missing data
            valid_data = merged_data.dropna(subset=["LapTime (s)"])
            
            if len(valid_data) > 0:
                # Use fixed model parameters
                n_estimators = 100
                learning_rate = 0.1
                
                X1_valid = valid_data[["QualifyingTime (s)"]]
                y1_valid = valid_data["LapTime (s)"]
                
                # Train models with fixed parameters
                X1_train, X1_test, y1_train, y1_test = train_test_split(
                    X1_valid, y1_valid, test_size=0.2, random_state=39)
                
                model1_fixed = GradientBoostingRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate, 
                    random_state=39)
                
                model1_fixed.fit(X1_train, y1_train)
                
                # Model 2 (with sector times)
                X2_valid = valid_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
                y2_valid = valid_data["LapTime (s)"]
                
                X2_train, X2_test, y2_train, y2_test = train_test_split(
                    X2_valid, y2_valid, test_size=0.2, random_state=38)
                
                model2_fixed = GradientBoostingRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate, 
                    random_state=38)
                
                model2_fixed.fit(X2_train, y2_train)
                
                # Make predictions using fixed models
                predictions1 = model1_fixed.predict(X1)
                
                # For model 2, we need to handle missing sector times
                X2 = merged_data[["QualifyingTime (s)", "Sector1Time (s)", 
                                 "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
                predictions2 = model2_fixed.predict(X2)
                
                # Add predictions to data
                merged_data["Model1_Prediction"] = predictions1
                merged_data["Model2_Prediction"] = predictions2
                
                # Display prediction for selected driver
                pred_cols = st.columns(2)
                
                with pred_cols[0]:
                    selected_pred1 = merged_data[merged_data["Driver_full"] == selected_driver]["Model1_Prediction"].values[0]
                    st.metric("Model 1 Prediction", f"{selected_pred1:.3f}s")
                    
                with pred_cols[1]:
                    selected_pred2 = merged_data[merged_data["Driver_full"] == selected_driver]["Model2_Prediction"].values[0]
                    st.metric("Model 2 Prediction", f"{selected_pred2:.3f}s", 
                             delta=f"{selected_pred2 - selected_pred1:.3f}s")
                
                # Visualizations
                st.header("Visual Summary")
                
                # Bar chart ranking all drivers
                st.subheader("Driver Rankings by Predicted Race Time")
                
                tab1, tab2 = st.tabs(["Model 1 (Qualifying Only)", "Model 2 (Qualifying + Sectors)"])
                
                with tab1:
                    chart_data1 = merged_data[["Driver_full", "QualifyingTime (s)", "Model1_Prediction"]].copy()
                    chart_data1 = chart_data1.sort_values(by="Model1_Prediction")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = sns.barplot(data=chart_data1, x="Model1_Prediction", y="Driver_full", ax=ax)
                    
                    # Highlight selected driver
                    for i, bar in enumerate(bars.patches):
                        if chart_data1.iloc[i]["Driver_full"] == selected_driver:
                            bar.set_color('red')
                    
                    plt.title("Drivers Ranked by Model 1 Predicted Race Time")
                    plt.xlabel("Predicted Race Time (s)")
                    plt.ylabel("Driver")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab2:
                    chart_data2 = merged_data[["Driver_full", "QualifyingTime (s)", "Model2_Prediction"]].copy()
                    chart_data2 = chart_data2.sort_values(by="Model2_Prediction")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = sns.barplot(data=chart_data2, x="Model2_Prediction", y="Driver_full", ax=ax)
                    
                    # Highlight selected driver
                    for i, bar in enumerate(bars.patches):
                        if chart_data2.iloc[i]["Driver_full"] == selected_driver:
                            bar.set_color('red')
                    
                    plt.title("Drivers Ranked by Model 2 Predicted Race Time")
                    plt.xlabel("Predicted Race Time (s)")
                    plt.ylabel("Driver")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Line chart comparing qualifying vs. predicted race time
                st.subheader("Qualifying Time vs. Predicted Race Time")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Sort by qualifying time
                compare_data = merged_data.sort_values(by="QualifyingTime (s)")
                
                plt.plot(compare_data["Driver_full"], compare_data["QualifyingTime (s)"], 
                         marker='o', label="Qualifying Time", linewidth=2)
                plt.plot(compare_data["Driver_full"], compare_data["Model1_Prediction"], 
                         marker='s', label="Model 1 Prediction", linewidth=2, linestyle='--')
                plt.plot(compare_data["Driver_full"], compare_data["Model2_Prediction"], 
                         marker='^', label="Model 2 Prediction", linewidth=2, linestyle=':')
                
                # Highlight selected driver
                selected_idx = compare_data[compare_data["Driver_full"] == selected_driver].index[0]
                plt.scatter(selected_driver, compare_data.loc[selected_idx, "QualifyingTime (s)"], 
                            color='red', s=100, zorder=5)
                plt.scatter(selected_driver, compare_data.loc[selected_idx, "Model1_Prediction"], 
                            color='red', s=100, zorder=5)
                plt.scatter(selected_driver, compare_data.loc[selected_idx, "Model2_Prediction"], 
                            color='red', s=100, zorder=5)
                
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.title("Qualifying Times vs. Predicted Race Times")
                plt.ylabel("Time (s)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature importance for model 2
                st.subheader("Feature Importance (Model 2)")
                
                importance = model2_fixed.feature_importances_
                feature_names = ["Qualifying Time", "Sector 1 Time", "Sector 2 Time", "Sector 3 Time"]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=importance, y=feature_names, ax=ax)
                plt.title(f"Feature Importance with n_estimators={n_estimators}, learning_rate={learning_rate}")
                plt.xlabel("Importance Score")
                plt.tight_layout()
                st.pyplot(fig)
            
            else:
                st.warning("Not enough valid data to train interactive models. Please ensure sector times are available.")
    
    except Exception as e:
        st.error(f"Error loading and processing data: {str(e)}")
        if "fastf1" in str(e).lower():
            st.info("If FastF1 is not installed, you can install it with: pip install fastf1")

    # Model code display (collapsible)
    with st.expander("View Model Code"):
        st.subheader("Model 1: Qualifying Time Only (prediction.py)")
        display_code("/Users/dhruviyer/F12025/prediction.py")
        
        st.subheader("Model 2: Qualifying + Sector Times (prediction2.py)")
        display_code("/Users/dhruviyer/F12025/prediction2.py")
        
        st.subheader("Model Comparison (f1_model_comparison.py)")
        display_code("/Users/dhruviyer/F12025/f1_model_comparison.py")

# Tab 3: Predictor (Original App)
with tab3:
    st.title("🏎️ F1 2025 Chinese GP Race Time Predictor")
    st.write("Predict race lap times using two Gradient Boosting models.")
    
    # Model descriptions
    st.subheader("About the Models")
    st.write("""
    - **Model 1: Qualifying Only**
      - Uses only the qualifying time to predict race lap times.
      - Trained on 2024 Australian GP data.
      - *Note*: Predictions may not be accurate for the Chinese GP due to track differences.

    - **Model 2: Qualifying + Sector Times**
      - Uses qualifying time and sector times from the 2024 Chinese GP.
      - More tailored to the Chinese GP with track-specific data.
      - *Note*: For new drivers, sector times are user-inputted and may affect accuracy.
    """)
    
    # Model and driver selection
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Select Model", ["Model 1: Qualifying Only", "Model 2: Qualifying + Sector Times"])
    with col2:
        all_drivers = ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
                   "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
                   "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
                   "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"]
        num_drivers = st.selectbox("Number of Drivers", options=list(range(1, len(all_drivers) + 1)), index=len(all_drivers) - 1)
    
    st.subheader("Select Drivers")
    selected_drivers = st.multiselect("Choose Drivers", options=all_drivers, default=all_drivers[:num_drivers])
    
    # Validate driver selection
    if len(selected_drivers) != num_drivers:
        st.warning(f"Please select exactly {num_drivers} drivers. Currently selected: {len(selected_drivers)}")
    else:
        # Input qualifying times
        st.header("Enter 2025 Chinese GP Qualifying Times")
        cols = st.columns(3)
        qualifying_times = {}
        for i, driver in enumerate(selected_drivers):
            with cols[i % 3]:
                qualifying_times[driver] = st.number_input(
                    f"{driver} Qualifying Time (s)",
                    min_value=85.0,
                    max_value=95.0,
                    value=90.0 - (i * 0.05),
                    step=0.001,
                    format="%.3f"
                )
    
        # Input sector times for Model 2
        sector1_times = {}
        sector2_times = {}
        sector3_times = {}
        if model_choice == "Model 2: Qualifying + Sector Times":
            st.header("Enter 2024 Chinese GP Average Sector Times")
            for driver in selected_drivers:
                st.subheader(f"🏎️ {driver}")
                cols = st.columns(3)
                with cols[0]:
                    sector1_times[driver] = st.number_input(
                        f"Sector 1 Time (s)",
                        min_value=20.0,
                        max_value=40.0,
                        value=28.0 + np.random.uniform(-0.5, 0.5),
                        step=0.001,
                        format="%.3f",
                        key=f"s1_{driver}"
                    )
                with cols[1]:
                    sector2_times[driver] = st.number_input(
                        f"Sector 2 Time (s)",
                        min_value=20.0,
                        max_value=40.0,
                        value=32.0 + np.random.uniform(-0.5, 0.5),
                        step=0.001,
                        format="%.3f",
                        key=f"s2_{driver}"
                    )
                with cols[2]:
                    sector3_times[driver] = st.number_input(
                        f"Sector 3 Time (s)",
                        min_value=20.0,
                        max_value=40.0,
                        value=30.0 + np.random.uniform(-0.5, 0.5),
                        step=0.001,
                        format="%.3f",
                        key=f"s3_{driver}"
                    )
    
        # Prediction button
        if st.button("🚀 Predict Race Times", type="primary"):
            # Prepare input data
            input_data = pd.DataFrame({
                "Driver": selected_drivers,
                "QualifyingTime (s)": [qualifying_times[driver] for driver in selected_drivers]
            })
            if model_choice == "Model 2: Qualifying + Sector Times":
                input_data["Sector1Time (s)"] = [sector1_times[driver] for driver in selected_drivers]
                input_data["Sector2Time (s)"] = [sector2_times[driver] for driver in selected_drivers]
                input_data["Sector3Time (s)"] = [sector3_times[driver] for driver in selected_drivers]
    
            # Make predictions
            if model_choice == "Model 1: Qualifying Only":
                predictions = model1.predict(input_data[["QualifyingTime (s)"]])
            else:
                predictions = model2.predict(input_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]])
            
            input_data["PredictedRaceTime (s)"] = predictions
            input_data = input_data.sort_values(by="PredictedRaceTime (s)")
            input_data["Gap to Leader (s)"] = input_data["PredictedRaceTime (s)"] - input_data["PredictedRaceTime (s)"].min()
    
            # Display results
            st.header("🏁 Predicted Race Times")
            display_df = input_data[["Driver", "PredictedRaceTime (s)", "Gap to Leader (s)"]]
            display_df["Position"] = range(1, len(display_df) + 1)
            display_df = display_df[["Position", "Driver", "PredictedRaceTime (s)", "Gap to Leader (s)"]]
            display_df["Gap to Leader (s)"] = display_df["Gap to Leader (s)"].apply(
                lambda x: f"+{x:.3f}s" if x > 0 else "Leader"
            )
            st.dataframe(
                display_df,
                column_config={
                    "Position": st.column_config.NumberColumn(format="%d"),
                    "PredictedRaceTime (s)": st.column_config.NumberColumn(format="%.3f"),
                },
                hide_index=True,
                use_container_width=True
            )
    
            # Visualization
            st.header("Visualization")
            col1, col2 = st.columns([3, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, max(6, len(selected_drivers) * 0.4)))
                colors = sns.color_palette("viridis", len(input_data))
                bars = sns.barplot(
                    data=input_data,
                    x="PredictedRaceTime (s)",
                    y="Driver",
                    palette=colors,
                    ax=ax
                )
                for i, p in enumerate(bars.patches):
                    width = p.get_width()
                    gap = input_data.iloc[i]["Gap to Leader (s)"]
                    gap_text = "LEADER" if gap == 0 else f"+{gap:.3f}s"
                    ax.text(
                        width + 0.1,
                        p.get_y() + p.get_height()/2,
                        gap_text,
                        ha='left',
                        va='center',
                        fontweight='bold' if gap == 0 else 'normal',
                        color='green' if gap == 0 else 'black'
                    )
                plt.title(f"Predicted 2025 Chinese GP Race Times ({model_choice})")
                plt.xlabel("Predicted Race Time (s)")
                plt.ylabel("Driver")
                plt.tight_layout()
                st.pyplot(fig)
    
            with col2:
                if model_choice == "Model 2: Qualifying + Sector Times":
                    st.subheader("Feature Importance")
                    importance = model2.feature_importances_
                    feature_names = ["Qualifying", "Sector 1", "Sector 2", "Sector 3"]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=importance, y=feature_names, palette="rocket", ax=ax)
                    plt.title("Model Feature Importance")
                    plt.xlabel("Importance Score")
                    plt.tight_layout()
                    st.pyplot(fig)