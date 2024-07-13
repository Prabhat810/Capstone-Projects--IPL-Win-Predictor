import streamlit as st
import pandas as pd
import pickle

# Declaring the Teams
teams = ['Royal Challengers Bangalore', 'Delhi Capitals', 'Mumbai Indians',
       'Kings XI Punjab', 'Kolkata Knight Riders', 'Sunrisers Hyderabad',
       'Rajasthan Royals', 'Chennai Super Kings']

# declaring the venues
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb'))

st.title("IPL Win Predictor")

col1, col2 = st.columns(2)

with col1:
    BattingTeam = st.selectbox('Select the Batting Team', sorted(teams))

with col2:
    BowlingTeam = st.selectbox('Select the Bowling Team', sorted(teams))

city = st.selectbox('Select the city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs = st.number_input('Overs Completed')

with col5:
    wickets = st.number_input('Wickets Fallen')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    currentrunrate = score/overs
    requiredrunrate = (runs_left*6)/balls_left

input_df = pd.DataFrame({'batting_team':[BattingTeam],'bowling_team':[BowlingTeam],'city':[city],'runs_left':[runs_left],
                         'balls_left':[balls_left],'wickets':[wickets],
                         'total_runs_x':[target],'current_run_rate':[currentrunrate],
                         'req_run_rate':[requiredrunrate]})

result = pipe.predict_proba(input_df)
lossprob = result[0][0]
winprob = result[0][1]

st.header(BattingTeam+" - "+str(round(winprob*100))+"%")

st.header(BowlingTeam+" - "+str(round(lossprob*100))+"%")

