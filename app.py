import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from PIL import Image 

st.set_page_config(page_title='Wild Blueberry Yield Prediction', page_icon = 'https://cdn-icons-png.flaticon.com/512/7696/7696639.png', layout='wide')
model=joblib.load(r'F:\Projects\Wild Blueberry yield Prediction\randomforest_blueberry_pollination_tuned_model.joblib')

st.markdown(
    """
    <style>
    [data-testid= "stAppViewContainer"] {
        
        background-image: url('https://img.freepik.com/free-vector/seamless-blueberry-pattern-pastel-background_53876-99173.jpg');
        background-size: cover; 
        
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    with st.form('prediction form'):

        st.title("Wild Blueberry Yield Prediction System")
        html_temp='''
        <div style='background-color:green; padding:12px'>
        <h1 style='color:  blue; text-align: center;'>Wild Blueberry yield Prediction Using Machine Learning </h1>
        </div>
        <h2 style='color:  green; text-align: center;'>Please Enter Input</h2>
    '''
        st.markdown(html_temp,unsafe_allow_html=True)

#        Min_Lower_TRange = st.number_input("Enter Min_Lower_TRange Value : ", value=None)
        clonesize = st.number_input("Enter Clonesize Value : ", value=None)
#        Average_Lower_TRange= st.number_input("Enter Average_Lower_TRange Value : ", value=None)
        bumbles = st.number_input("Enter Bumbles Value : ", value=None)
#        Average_Upper_TRange = st.number_input("Enter Average_Upper_TRange Value : ", value=None)
        andrena = st.number_input("Enter andrena Value : ", value=None)
#        Average_Raining_Days = st.number_input("Enter Average_Raining_Days Value : ", value=None)
        osmia = st.number_input("Enter osmia Value : ", value=None)
        Max_Upper_TRange= st.number_input("Enter Max_Upper_TRange Value : ", value=None)
        Raining_Days = st.number_input("Enter Raining_Days Value : ", value=None)
        fruit_set = st.number_input("Enter fruit_set Value : ", value=None)

        seeds= st.number_input("Enter seeds Value : ", value=None)
#        Max_Lower_TRange= st.number_input("Enter Max_Lower_TRange Value : ", value=None)
#        honeybee = st.number_input("Enter honeybee value : ", value=None)
#        Min_Upper_TRange = st.number_input("Enter Min_Upper_TRange value: ", value = None)
        fruit_mass = st.number_input("Enter fruit_mass value : ", value = None)
        submit = st.form_submit_button()

    if submit:
        data  = np.array([float(clonesize), float(bumbles), float(andrena), float(osmia), 
                            float(Max_Upper_TRange), float(Raining_Days), float(fruit_set), float(seeds),  float(fruit_mass)]).reshape(1,-1)

        pred = get_prediction(data = data)
        st.write(f"The blueberry yield is : {pred}")
        st.balloons()
def get_prediction(data):
    model_path = r"F:\Projects\Wild Blueberry yield Prediction\randomforest_blueberry_pollination_tuned_model.joblib"
    model = joblib.load(model_path)
    return model.predict(data)


if __name__ == '__main__':
    main()