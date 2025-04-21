import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="F1 2025 Chinese GP Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Streamlit app
st.title("🏎️ F1 2025 Chinese GP Race Time Predictor")
st.write("Predict race lap times using two Gradient Boosting models: one with qualifying times only, and one with qualifying and sector times.")

# Load or create models
model1 = None
model2 = None

try:
    # Try to load existing models
    if os.path.exists("model1_gb.pkl"):
        with open("model1_gb.pkl", "rb") as f:
            model1 = pickle.load(f)
        st.success("Model 1 loaded successfully!")
    
    if os.path.exists("model2_gb.pkl"):
        with open("model2_gb.pkl", "rb") as f:
            model2 = pickle.load(f)
        st.success("Model 2 loaded successfully!")
except Exception as e:
    st.warning(f"Error loading models: {e}")

# Create placeholder models if they don't exist
if model1 is None:
    st.info("Model 1 not found. Creating a placeholder model...")
    # Create a simple model that predicts race time = qualifying time * factor
    model1 = GradientBoostingRegressor(n_estimators=10, random_state=39)
    # Simple training data
    X_train = np.array([[85.0], [87.0], [89.0], [91.0], [93.0]])
    y_train = np.array([85.0 * 1.05, 87.0 * 1.05, 89.0 * 1.05, 91.0 * 1.05, 93.0 * 1.05])
    model1.fit(X_train, y_train)
    
    # Save the model for future use
    with open("model1_gb.pkl", "wb") as f:
        pickle.dump(model1, f)
    st.success("Created and saved placeholder Model 1!")

if model2 is None:
    st.info("Model 2 not found. Creating a placeholder model...")
    # Create a more complex model that uses qualifying time and sector times
    model2 = GradientBoostingRegressor(n_estimators=10, random_state=38)
    # Simple training data with 4 features (qualifying + 3 sectors)
    X_train = np.array([
        [85.0, 25.0, 30.0, 30.0],
        [87.0, 26.0, 31.0, 30.0],
        [89.0, 27.0, 31.0, 31.0],
        [91.0, 28.0, 32.0, 31.0],
        [93.0, 29.0, 33.0, 31.0]
    ])
    y_train = np.array([85.0 * 1.04, 87.0 * 1.04, 89.0 * 1.04, 91.0 * 1.04, 93.0 * 1.04])
    model2.fit(X_train, y_train)
    
    # Save the model for future use
    with open("model2_gb.pkl", "wb") as f:
        pickle.dump(model2, f)
    st.success("Created and saved placeholder Model 2!")

# Model selection
col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox("Select Model", ["Model 1: Qualifying Only", "Model 2: Qualifying + Sector Times"])

# Driver list for 2025
all_drivers = ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
           "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
           "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
           "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"]

# Driver selection options
with col2:
    num_drivers = st.selectbox("Number of Drivers to Include", 
                            options=list(range(1, len(all_drivers) + 1)), 
                            index=len(all_drivers) - 1)  # Default to all drivers

st.subheader("Select Drivers")
selected_drivers = st.multiselect("Choose Drivers", 
                                options=all_drivers,
                                default=all_drivers[:num_drivers])

# Ensure number of selected drivers matches the chosen number
if len(selected_drivers) != num_drivers:
    st.warning(f"Please select exactly {num_drivers} drivers. Currently selected: {len(selected_drivers)}")
else:
    # Input qualifying times
    st.header("Enter 2025 Chinese GP Qualifying Times")
    
    # Create columns for more compact layout
    cols = st.columns(3)
    qualifying_times = {}
    for i, driver in enumerate(selected_drivers):
        with cols[i % 3]:
            qualifying_times[driver] = st.number_input(
                f"{driver} Qualifying Time (s)", 
                min_value=85.0, 
                max_value=95.0, 
                value=90.0 - (i * 0.05),  # Slightly different default values
                step=0.001,
                format="%.3f"
            )

    # Input sector times (for Model 2)
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
                    value=28.0 + np.random.uniform(-0.5, 0.5),  # Random default values
                    step=0.001,
                    format="%.3f",
                    key=f"s1_{driver}"
                )
            with cols[1]:
                sector2_times[driver] = st.number_input(
                    f"Sector 2 Time (s)", 
                    min_value=20.0, 
                    max_value=40.0, 
                    value=32.0 + np.random.uniform(-0.5, 0.5),  # Random default values
                    step=0.001,
                    format="%.3f",
                    key=f"s2_{driver}"
                )
            with cols[2]:
                sector3_times[driver] = st.number_input(
                    f"Sector 3 Time (s)", 
                    min_value=20.0, 
                    max_value=40.0, 
                    value=30.0 + np.random.uniform(-0.5, 0.5),  # Random default values
                    step=0.001,
                    format="%.3f",
                    key=f"s3_{driver}"
                )

    if st.button("🚀 Predict Race Times", type="primary"):
        # Create input DataFrame
        input_data = pd.DataFrame({
            "Driver": selected_drivers,
            "QualifyingTime (s)": [qualifying_times[driver] for driver in selected_drivers]
        })

        # Add sector times for Model 2
        if model_choice == "Model 2: Qualifying + Sector Times":
            input_data["Sector1Time (s)"] = [sector1_times[driver] for driver in selected_drivers]
            input_data["Sector2Time (s)"] = [sector2_times[driver] for driver in selected_drivers]
            input_data["Sector3Time (s)"] = [sector3_times[driver] for driver in selected_drivers]

        # Predict
        if model_choice == "Model 1: Qualifying Only":
            predictions = model1.predict(input_data[["QualifyingTime (s)"]])
        else:
            predictions = model2.predict(input_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]])
        
        input_data["PredictedRaceTime (s)"] = predictions
        input_data = input_data.sort_values(by="PredictedRaceTime (s)")
        
        # Calculate time differences
        fastest_time = input_data["PredictedRaceTime (s)"].min()
        input_data["Gap to Leader (s)"] = input_data["PredictedRaceTime (s)"] - fastest_time
        
        # Display results
        st.header("🏁 Predicted Race Times")
        
        # Format for display
        display_df = input_data[["Driver", "PredictedRaceTime (s)", "Gap to Leader (s)"]]
        display_df["Position"] = range(1, len(display_df) + 1)
        display_df = display_df[["Position", "Driver", "PredictedRaceTime (s)", "Gap to Leader (s)"]]
        
        # Format the gap column
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
            
            # Create custom color palette based on position
            colors = sns.color_palette("viridis", len(input_data))
            
            # Create bar plot
            bars = sns.barplot(
                data=input_data, 
                x="PredictedRaceTime (s)", 
                y="Driver", 
                palette=colors,
                ax=ax
            )
            
            # Add time gap annotations
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
            # Show feature importance if using Model 2
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