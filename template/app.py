import pandas as pd
import streamlit as st
import detection
from streamlit_option_menu import option_menu
import streamlit_toggle as tog
import functionalities

st.set_page_config(page_title="Disease Detection Application", page_icon="https://img.icons8.com/color/48/null/caduceus.png", layout="wide")
with st.sidebar:
    choice = option_menu('Multiple Disease Detection',['Chronic Kidney Disease','Parkinson\'s Disease'])

if choice == 'Chronic Kidney Disease':
    st.title("Chronic Kidney Disease Detection")

    fileStatus, file = functionalities.load_file()
    if fileStatus:
        requirement = st.selectbox("",['Prediction','Analysis Report'])
        functionalities.action(requirement, file, "chronic")

if choice == 'Parkinson\'s Disease':
    st.title('Parkinson\'s Disease Detection')

    option = st.selectbox("Select an option", ["Analysis Using MDVP", "Analysis Using UPDRS"])
    
    if option == 'Analysis Using MDVP':

        fileStatus, file = functionalities.load_file()
        if fileStatus:
            requirement = st.selectbox("",['Prediction','Analysis Report'])
            functionalities.action(requirement, file, "parkinsons")
    
    if option == 'Analysis Using UPDRS':
        fileStatus, file = functionalities.load_file()
        if fileStatus:
            requirement = st.selectbox("",['Prediction','Analysis Report'])
            functionalities.action(requirement, file ,"parkinsons_udprs")