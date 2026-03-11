
import streamlit as st
import pickle
import pandas as pd

# --- 1. Load the Model ---
# We load the pickle file we just created in the Jupyter Notebook
pipe = pickle.load(open('pipe.pkl', 'rb'))

# --- 2. Define the UI Data ---
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# --- 3. Build the UI Layout ---
st.title('IPL Win Probability Predictor 🏏')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

selected_city = st.selectbox('Select Host City', sorted(cities))

target = st.number_input('Target Score (1st Innings Total + 1)', step=1, min_value=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', step=1, min_value=0)
with col4:
    overs = st.number_input('Overs Completed', step=0.1, min_value=0.0, max_value=20.0)
with col5:
    wickets = st.number_input('Wickets Down', step=1, min_value=0, max_value=10)

# --- 4. Prediction Logic ---
if st.button('Predict Probability'):
    # Prevent division by zero and invalid states
    if overs == 0:
        st.error("Please enter overs > 0 to calculate run rates.")
    elif score > target:
        st.success(f"{batting_team} has already won!")
    elif wickets == 10 and score < target:
        st.error(f"{bowling_team} has already won!")
    else:
        # Calculate dynamic features
        runs_left = target - score
        balls_left = 120 - int(overs * 6) # Rough estimation for simplicity in UI
        wickets_left = 10 - wickets
        crr = (score * 6) / (120 - balls_left)
        rrr = (runs_left * 6) / balls_left

        # Create input dataframe matching the exact columns the model expects
        input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                                 'city': [selected_city], 'runs_left': [runs_left], 
                                 'balls_left': [balls_left], 'wickets_left': [wickets_left], 
                                 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

        # Get probabilities
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display results nicely
        st.header(f"{batting_team} Win Probability: {round(win * 100, 1)}%")
        st.header(f"{bowling_team} Win Probability: {round(loss * 100, 1)}%")
