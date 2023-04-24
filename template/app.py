import pandas as pd
import streamlit as st
import detection
from streamlit_option_menu import option_menu
import streamlit_toggle as tog
import functionalities

st.set_page_config(page_title="Disease Detection Application", page_icon="https://img.icons8.com/color/48/null/caduceus.png")

with st.sidebar:
    choice = option_menu('Multiple Disease Detection',['Chronic Kidney Disease','Parkinson\'s Disease'])

if choice == 'Chronic Kidney Disease':
    st.title("Chronic Kidney Disease Detection")

    option = st.selectbox("Select an option", ["Enter the data", "Analysis"])

    if option == 'Enter the data':
        default_value = ''
        st.markdown("If RBC and pc are normal: Toggle swith on .\n\n If pcc and ba are present: Toggle switch on.\n\n If htn, cad, dm, pe and ane are no: Toggle switch off. \n\n If appet is good: Toggle Switch off.")
        with st.form("my_form"):
            col1, col2, col3, col4 = st.columns(4)
        with col1:
            Name =st.text_input('Name:')
            if Name.isalpha():
                pass
            else:
                st.write("Enter the valid name")
            Age = st.text_input('Age')
            if Age.isnumeric():
                try:
                    if int(Age) <= 0:
                        st.write("Please enter valid input")
                except ValueError:
                    st.write("Invalid input. Please enter valid number")
                pass
            else:
                st.write("Please enter a valid age")
            bp = st.text_input('Blood Pressure(bp)', default_value)
            sg = st.text_input('Specific Gravity(sg)', default_value)
            al = st.text_input('Albumin(al)', default_value)
            su = st.text_input('Sugar(su)', default_value)
        with col2:
            bgr = st.text_input('Blood Glucose Random(bgr)', default_value)
            bu = st.text_input('Blood Urea(bu)', default_value)
            sc = st.text_input('Serum Creatinine(sc)', default_value)
            sod = st.text_input('Sodium(sod)', default_value)
            pot = st.text_input('Potassium(pot)', default_value)
            hemo = st.text_input('Hemoglobin(hemo)', default_value)
        with col3:
            pcv = st.text_input('Packed Cell Volume(pcv)', default_value)
            wc = st.text_input('White Blood Cell Count(wc)', default_value)
            rc = st.text_input('Red Blood Cell Count(rc)', default_value)
            rbc_switch = tog.st_toggle_switch(label="Red Blood Cell(rbc)", 
                        key="Key1", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if rbc_switch:
                rbc = 1
            else:
                rbc = 0
            pc_switch = tog.st_toggle_switch(label="Pus Cell(pc)", 
                        key="Key2", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if pc_switch:
                pc = 1
            else:
                pc = 0
            pcc_switch = tog.st_toggle_switch(label="Pus Cell Clumps(pcc)", 
                        key="Key3", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if pcc_switch:
                pcc = 1
            else:
                pcc = 0
            ba_switch = tog.st_toggle_switch(label="Bacteria(ba)", 
                        key="Key4", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if ba_switch:
                ba = 1
            else:
                ba = 0
        with col4:
            htn_switch = tog.st_toggle_switch(label="Hypertension(htn)", 
                        key="Key5", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if htn_switch:
                htn = 1
            else:
                htn = 0
            dm_switch = tog.st_toggle_switch(label="Diabetes Mellitus(dm)", 
                        key="Key6", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if dm_switch:
                dm = 1
            else:
                dm = 0
            cad_switch = tog.st_toggle_switch(label="Coronary Artery Disease(cad)", 
                        key="Key7", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if cad_switch:
                cad = 1
            else:
                cad = 0
            appet_switch = tog.st_toggle_switch(label="Appetitie(appet)", 
                        key="Key8", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if appet_switch:
                appet = 1
            else:
                appet = 0
            pe_switch = tog.st_toggle_switch(label="Pedal Edema(pe)", 
                        key="Key9", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if pe_switch:
                pe = 1
            else:
                pe = 0
            ane_switch = tog.st_toggle_switch(label="Anemia(ane)", 
                        key="Key10", 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
            if ane_switch:
                ane = 1
            else:
                ane = 0
        with col1:
            form_button = st.form_submit_button(label='Predict Disease')
        diagnosis = ''
        if all([
            bp.replace('-', '').replace('.', '').isdigit(), 
            sg.replace('-', '').replace('.', '').isdigit(),
            al.replace('-', '').replace('.', '').isdigit(),
            su.replace('-', '').replace('.', '').isdigit(),
            bgr.replace('-', '').replace('.', '').isdigit(),
            bu.replace('-', '').replace('.', '').isdigit(),
            sc.replace('-', '').replace('.', '').isdigit(),
            sod.replace('-', '').replace('.', '').isdigit(),
            pot.replace('-', '').replace('.', '').isdigit(),
            hemo.replace('-', '').replace('.', '').isdigit(),
            pcv.replace('-', '').replace('.', '').isdigit(),
            wc.replace('-', '').replace('.', '').isdigit(),
            rc.replace('-', '').replace('.', '').isdigit()]):
            if form_button:
                diagnosis = detection.chronic_detection([Age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane])
                if diagnosis[0] == 0:
                    st.success("The person dose not have Chronic Kidney Disease")
                else:
                    st.error("The person has Chronic Kidney Disease")
        else:
            st.write("Please enter valid Input")
    
    if option == 'Analysis':
        fileStatus, file = functionalities.load_file()
        if fileStatus:
            requirement = st.selectbox("",['Prediction','Analysis Report'])
            functionalities.action(requirement, file, "chronic")

if choice == 'Parkinson\'s Disease':
    st.title('Parkinson\'s Disease Detection')

    option = st.selectbox("Select an option", ["Enter the data", "Analysis Using MDVP", "Analysis Using UPDRS"])

    if option == 'Enter the data':
        default_value = ''
        with st.form("my_form"):
            col1, col2, col3, col4 = st.columns(4)

        with col1:
            Name =st.text_input('Name:')
            if Name.isalpha():
                pass
            else:
                st.write("Enter the valid name")
        with col2:
            Age = st.text_input('Age')
            if Age.isnumeric():
                try:
                    if int(Age) <= 0:
                        st.write("Please enter valid input")
                except ValueError:
                    st.write("Invalid input. Please enter valid number")
                pass
            else:
                st.write("Please enter a valid age")
        with col3:
            fo = st.text_input('MDVP: Fo(Hz)', default_value)
        with col4:
            fhi = st.text_input('MDVP: Fhi(Hz)', default_value)
        with col1:
            flo = st.text_input('MDVP: Flo(Hz)', default_value)
        with col2:
            Jitter_percent = st.text_input('MDVP: Jitter(%)', default_value)
        with col3:
            Jitter_Abs = st.text_input('MDVP: Jitter(Abs)', default_value)
        with col4:
            RAP = st.text_input('MDVP: RAP', default_value)
        with col1:
            PPQ = st.text_input('MDVP: PPQ', default_value)
        with col2:
            DDP = st.text_input('Jitter: DDP', default_value)
        with col3:
            Shimmer = st.text_input('MDVP: Shimmer', default_value)
        with col4:
            Shimmer_dB = st.text_input('MDVP: Shimmer(dB)', default_value)
        with col1:
            APQ3 = st.text_input('Shimmer: APQ3', default_value)
        with col2:
            APQ5 = st.text_input('Shimmer: APQ5', default_value)
        with col3:
            APQ = st.text_input('MDVP: APQ', default_value)
        with col4:
            DDA = st.text_input('Shimmer: DDA', default_value)
        with col1:
            NHR = st.text_input('NHR', default_value)
        with col2:
            HNR = st.text_input('HNR', default_value)
        with col3:
            RPDE = st.text_input('RPDE', default_value)
        with col4:
            DFA = st.text_input('DFA', default_value)
        with col1:
            spread1 = st.text_input('spread1', default_value)
        with col2:
            spread2 = st.text_input('spread2', default_value)
        with col3:
            D2 = st.text_input('D2', default_value)
        with col4:
            PPE = st.text_input('PPE', default_value)
        with col1:
            form_button = st.form_submit_button(label='Predict Disease')

        diagnosis = ''
        if all([fo.replace('-', '').replace('.', '').isdigit(), 
                fhi.replace('-', '').replace('.', '').isdigit(),
                flo.replace('-', '').replace('.', '').isdigit(),
                Jitter_percent.replace('-', '').replace('.', '').isdigit(),
                Jitter_Abs.replace('-', '').replace('.', '').isdigit(),
                RAP.replace('-', '').replace('.', '').isdigit(),
                PPQ.replace('-', '').replace('.', '').isdigit(),
                DDP.replace('-', '').replace('.', '').isdigit(),
                Shimmer.replace('-', '').replace('.', '').isdigit(),
                Shimmer_dB.replace('-', '').replace('.', '').isdigit(),
                APQ3.replace('-', '').replace('.', '').isdigit(),
                APQ5.replace('-', '').replace('.', '').isdigit(),
                APQ.replace('-', '').replace('.', '').isdigit(),
                DDA.replace('-', '').replace('.', '').isdigit(),
                NHR.replace('-', '').replace('.', '').isdigit(),
                HNR.replace('-', '').replace('.', '').isdigit(),
                RPDE.replace('-', '').replace('.', '').isdigit(),
                DFA.replace('-', '').replace('.', '').isdigit(),
                spread1.replace('-', '').replace('.', '').isdigit(),
                spread2.replace('-', '').replace('.', '').isdigit(),
                D2.replace('-', '').replace('.', '').isdigit(),
                PPE.replace('-', '').replace('.', '').isdigit()]):
            if form_button:
                diagnosis = detection.parkinsons_prediction([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])
                if diagnosis[0] == 0:
                    st.success("The person dose not have Parkinson's Disease")
                else:
                    st.error("The person has Parkinson's Disease")
        
        else:
            st.write("Please enter valid Input")
    
    if option == 'Analysis Using MDVP':

        fileStatus, file = functionalities.load_file()
        if fileStatus:
            requirement = st.selectbox("",['Prediction','Analysis Report'])
            functionalities.action(requirement, file, "parkinsons")
    
    if option == 'Analysis Using UPDRS':
        fileStatus, file = functionalities.load_file()
        if fileStatus:
            functionalities.analysisReportUPDRS(file)

