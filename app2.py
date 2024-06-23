import numpy as np
import joblib
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)


pipe4 = joblib.load("loan_status_prediction.pkl")

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_loan_status_prediction(Age,Experience,Income,Family,CCAvg,Education,Mortgage,security_account,cd_account,Online,CreditCard):
    
   
    prediction=pipe4.predict([[Age,Experience,Income,Family,CCAvg,Education,Mortgage,security_account,cd_account,Online,CreditCard]])
    print(prediction)
    return prediction



def main():
    st.title("Loan Status Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Age = st.text_input("Age","Type Here")
    Experience = st.text_input("Experience","Type Here")
    Income = st.text_input("Income","Type Here")
    Family = st.text_input("Family","Type Here")
    CCAvg = st.text_input("CCAvg","Type Here")
    Education = st.text_input("Education","Type Here")
    Mortgage = st.text_input("Mortgage","Type Here")
    security_account = st.text_input("security_account","Type Here")
    cd_account = st.text_input("cd_account","Type Here")
    Online = st.text_input("Online","Type Here")
    CreditCard = st.text_input("CreditCard","Type Here")
    


    result=""
    if st.button("Predict"):
        result=predict_loan_status_prediction(Age,Experience,Income,Family,CCAvg,Education,Mortgage,security_account,cd_account,Online,CreditCard)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()