#DL-App for sentiment analysis - hotel reviews!!

#Import Libraries
import pandas as pd
import streamlit as st 
from pickle import load 
from PIL import Image
import time
from streamlit_lottie import st_lottie, st_lottie_spinner
import json
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification

#Background config
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
lottie_start = load_lottiefile("38825-robot-hello.json")
lottie_type  = load_lottiefile("111084-brain-yeloow (1).json")
lottie_process = load_lottiefile("72689-brain-bulb-with-gears.json")


#Input variables 
def user_input_features():
    X = st.text_area(label='Play around with our sentiment analyzer:', placeholder=('Type your text here and check the sentiment!!'))

    return X

#Loading the model
model  = pipeline("sentiment-analysis", model = 'D:/Excel_R/Data Science/Projects/NLP/Project_P155_Hotelreviews/mod1')


#Home Webage
if not st.checkbox("Confirm you're not ROBOT!!"):
    st_lottie(lottie_start, height=400, width=None, quality="high", speed=1.12, loop=True)

#Sentiment Analysis Page
else:

#Title of web page
    st.title("Sentiment Analyzer")

    st_lottie(lottie_type, height=207, width=195, quality="high", speed=1.4, loop=True)

    df = user_input_features()

    result = model(df)

    dg = pd.DataFrame(result[0], index=['Result'])

    dg['score'] = dg['score']*100

    dg.rename(columns={"label":"Sentiment", "score":"Score"})

#Predicting the result
    if st.button('Classify Text'):
        st.subheader('Predicted Sentiment:')
        with st_lottie_spinner(lottie_process, height=(225), width=(700), quality="low", speed=1.25):
            time.sleep(3)
            dg

#Running the function
if __name__=='__user_input_features__':
    user_input_features()   

