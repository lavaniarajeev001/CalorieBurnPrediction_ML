import streamlit as st
import pandas as pd
import numpy as np
import pickle
from src.pipeline import predict
from src.utils import load_object
import os
from src.utils import get_encoded_values
def add_sidebar():
    train_df=pd.read_csv("artifacts\\train.csv").drop(["Calories_Burned"],axis=1)
    test_df=pd.read_csv("artifacts\\test.csv").drop(["Calories_Burned"],axis=1)
    st.header("Attributes")
    slider_label = [("Age", "Age"),("Weight (kg)", "Weight (kg)"),("Height (m)", "Height (m)"),("Max_BPM", "Max_BPM"),
                    ("Avg_BPM", "Avg_BPM"),("Resting_BPM", "Resting_BPM"),
                    ("Session_Duration (hours)", "Session_Duration (hours)"),
                    ("Fat_Percentage", "Fat_Percentage"),("Water_Intake (liters)", "Water_Intake (liters)"),
                    ("Workout_Frequency (days/week)", "Workout_Frequency (days/week)"),("Experience Level","Experience_Level"),
                    ("BMI", "BMI")
                    ]
    
    input_dict = {}
    for label, key in slider_label:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0,
            max_value=int(train_df[key].max())
        )
    input_dict["Gender"] = st.sidebar.selectbox(
        "Gender",
        options=train_df["Gender"].unique()  # Dynamic options from the dataset
    )
    input_dict["Workout_Type"] = st.sidebar.selectbox(
        "Workout Type",
        options=train_df["Workout_Type"].unique()  # Dynamic options from the dataset
    )

    input_df = pd.DataFrame([input_dict])
    return input_df
    

def add_prediction(input_data):
    
    # Load the saved model and scalers
    model_path=os.path.join("artifacts","model.pkl")
    preprocessing_path=os.path.join("artifacts","preprocessing.pkl")
    model=load_object(file_path=model_path)
    preprocessor=load_object(file_path=preprocessing_path)

    
    # Scale the input features using scaler_X
    #input_scaled = scaler_X.transform(input_array)
    
    # Make the prediction
    columns=[
    "Age", "Gender", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM",
    "Resting_BPM", "Session_Duration (hours)", "Workout_Type",
    "Fat_Percentage", "Water_Intake (liters)", "Workout_Frequency (days/week)",
    "Experience_Level", "BMI"]
    data_scaled = preprocessor.transform(input_data) 
    preds=model.predict(data_scaled)
    
    # Display the prediction
    st.subheader("Prediction")
    st.write("The total calories burned are:") 
    st.write(f"{preds[0]:.2f}")  # Displaying with two decimal places

def main():
    st.set_page_config(
        page_title="Calorie Burn App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Collect user input from sidebar
    input_data = add_sidebar()
    
    # Main container for the app
    with st.container():
        st.title("Calories Burn App")
        st.write("This app is designed for the prediction of calories burned based on the provided attributes.")
        
    # Trigger prediction when the button is clicked
    if st.button("Predict"):
        add_prediction(input_data)

if __name__ == "__main__":
    main()