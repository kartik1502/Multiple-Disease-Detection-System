import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_toggle as tog

parkinsons_model = pickle.load(open('C:/Users/karti/Disease Detection/models/trained_model_parkinsons.sav', 'rb'))

chronic_model = pickle.load(open('C:/Users/karti/Disease Detection/models/trained_model_chronic_kidney.sav','rb'))

def parkinsons_prediction(input_data):
    input_data = np.asarray(input_data)

    input_data_reshaped = input_data.reshape(1, -1)

    result = parkinsons_model.predict(input_data_reshaped)

    if result[0] == 0:
        return "The person dose not have Parkinson's Disease"
    else:
        return "The person has Parkinson's Disease"

def chronic_prediction(input_data):
    input_data = np.asarray(input_data, dtype=np.float64)

    input_data_reshaped = input_data.reshape(1, -1)
    
    result = chronic_model.predict(input_data_reshaped)

    if result[0] == 0:
        return "The person is diseased with Chronic Kidney disease"
    else:
        return "The person is not diseased with Chronic Kidney disease"

    
from io import BytesIO
from reportlab.pdfgen import canvas

def generate_report(diagnosis, input_data):
    buffer = BytesIO()  
    pdf = canvas.Canvas(buffer)  
    pdf.setTitle("Parkinson's Disease Prediction Report")  
    pdf.setFont("Helvetica", 12)  
    pdf.drawString(50, 750, "Parkinson's Disease Prediction Report")
    pdf.drawString(50, 700, f"Diagnosis: {diagnosis}")
    y = 650 
    for key, value in input_data.items():
        pdf.drawString(50, y, f"{key}: {value}")
        y -= 25
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer
st.set_page_config(page_icon="https://img.icons8.com/fluency/48/null/caduceus.png", page_title="Multiple Disease Detection", layout='wide')
with st.sidebar:
    choice = option_menu('Multiple Disease Detection',['Parkinson\'s Disease','Chronic Kidney Disease'])

if choice == 'Parkinson\'s Disease':
    
    st.title("Parkinson's Disease Detection")
    
    options = ["Enter Data", "Choose file"]
    selected_option = st.selectbox("Select an option", options)
    
    if selected_option == 'Enter Data':
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

        input_data = {
                'MDVP: Fo(Hz)': fo,
                'MDVP: Fhi(Hz)': fhi,
                'MDVP: Flo(Hz)': flo,
                'MDVP: Jitter(%)': Jitter_percent,
                'MDVP: Jitter(Abs)': Jitter_Abs,
                'MDVP: RAP': RAP,
                'MDVP: PPQ': PPQ,
                'Jitter: DDP': DDP, 
                'MDVP: Shimmer': Shimmer,
                'MDVP: Shimmer(dB)': Shimmer_dB,
                'Shimmer: APQ3': APQ3,
                'Shimmer: APQ5': APQ5,
                'MDVP: APQ': APQ,
                'Shimmer: DDA': DDA,
                'NHR': NHR,
                'HNR': HNR,
                'RPDE': RPDE,
                'DFA': DFA,
                'spread1': spread1,
                'spread2': spread2,
                'D2': D2,
                'PPE': PPE
            }

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
                diagnosis = parkinsons_prediction([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])
                st.success(diagnosis)
                # report_generate(Name, Age, input_data, diagnosis)
                pdf_buffer = generate_report(diagnosis, input_data)
                # st.download_button(label="Download Report", data=pdf_buffer.getvalue(), file_name=f"{Name}.pdf", mime="application/pdf")
        
        else:
            st.write("Please enter valid Input")
    
    if selected_option == 'Choose file':
        upload_file = st.file_uploader("choose a file")
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            df['result'] = 'Not diseased'
            for index, row in df.iterrows():
                # st.write(f"Row: {list(row)}") 
                input_data = np.asarray(list(row[:-1]), dtype=np.float64)
                input_data_reshaped = input_data.reshape(1, -1)
                result = parkinsons_model.predict(input_data_reshaped)
                if result[0] == 0:
                    df.at[index, 'result'] = 'Not diseased'
                else:
                    df.at[index, 'result'] = 'Diseased'
            st.write(df)
            df.to_csv('updated_data.csv', index=False)

        # Add a download button to download the CSV file
            st.download_button(
                label='Download Updated Data',
                data=df.to_csv().encode('utf-8'),
                file_name='updated_data.csv',
                mime='text/csv'
            )

    

if choice == 'Chronic Kidney Disease':
    st.title("Chronic Kidney Disease")
    
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

    input_data = {
        'bp': bp,
        'sg': sg,
        'al': al,
        'su': su,
        'rbc': rbc,
        'pc': pc,
        'pcc': pcc,
        'ba': ba,
        'bgr': bgr,
        'bu': bu,
        'sc': sc,
        'sod': sod,
        'pot': pot,
        'hemo': hemo,
        'pcv': pcv,
        'wc': wc,
        'rc': rc,
        'htn': htn,
        'dm': dm,
        'cad': cad,
        'appet': appet,
        'pe': pe,
        'ane': ane
    }

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
            diagnosis = chronic_prediction([Age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane])
            st.success(diagnosis)
            pdf_buffer = generate_report(diagnosis, input_data)
            # st.download_button(label="Download Report", data=pdf_buffer.getvalue(), file_name=f"{Name}.pdf", mime="application/pdf")
    
    else:
        st.write("Please enter valid Input")
