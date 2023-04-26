import streamlit as st
import pandas as pd
import detection
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_file():
    loaded_file = st.file_uploader("")
    if loaded_file is None:
        return False, loaded_file
    else:
        return True, loaded_file
    
def displayModelResult(trainAccuracy, testAccuracy, yTrain, yTest, trainPreds, testPreds, model):
    st.write("Accuracy on the Train Data by", model,":", trainAccuracy)
    st.markdown("Confusion Matrix on Train data for "+model )
    detection.displayConfusioMatrix(yTrain, trainPreds)
    st.write("Accuracy on the Test Data by", model,":", testAccuracy)
    st.markdown("Confusion Matrix on Test data "+model)
    detection.displayConfusioMatrix(yTest, testPreds)
    st.write(model +" Classification Report")
    st.text(detection.classificationReport(yTest, testPreds))

def displayKFoldResult(model, scores, mean):
    st.markdown(model+" CV Scores:")
    st.write(scores)
    st.markdown(model+" CV Mean Score:")
    st.write(mean)

def action(requirement, file, disease):
    show_info = st.checkbox('View Sample Data', value=False)
    df = pd.read_csv(file)
    if disease == 'parkinsons_udprs':
        targetVariable = st.radio("Select the target variable",("Motor UDPRS","Total UDPRS"))
        if targetVariable is None:
            st.error("Please enter the target variable")
        if targetVariable == 'Motor UDPRS':
            bins = [0, 20, float('inf')]
            targetVariable = 'motor_UPDRS'
            variables = 'total_UPDRS'
        else:
            targetVariable = 'total_UPDRS'
            bins = [0, 25, float('inf')]
            variables = 'motor_UPDRS'
    if disease != 'parkinsons_udprs':
        targetVariable = st.text_input("Enter the target variable")
        if targetVariable is None:
            st.error("Please enter the target variable")
    if targetVariable not in df.columns:
        st.error("The provided target variable is not present in the dataset")
    else:
        if show_info:
            st.write(df.drop(targetVariable, axis=1).head())
        variable = st.text_input("Enter the variable that has to ignored in the prediction")
        flag = True
        labelEncoderStatus = False
        imputerStatus = False
        if variable != "" and variable not in df.columns:
            st.write(variable)
            st.error("The provided variable is not present in the dataset")
            flag = False
        elif(variable != ""):
            col_indexs = df.columns.get_loc(variable)
            col_removed = df[variable]
            df = df.drop(variable, axis=1)
            for column in df.columns:
                if df[column].dtype == 'object':
                    labelEncoderStatus = True
            if labelEncoderStatus:
                labelEncoder = st.checkbox('Apply Label Encoder', value=False)
                if labelEncoder:
                    for column in df.columns:
                        if df[column].dtype == 'object':
                            le = LabelEncoder()
                            df[column] = le.fit_transform(df[column])
                            labelEncoderStatus = False
                    if show_info:
                        st.write(df.drop(targetVariable, axis=1).head())
            if df.isna().any().any():
                imputerStatus = True
                imputer = st.checkbox('Apply Imputer', value=False)
                if imputer:
                    imputer = SimpleImputer(strategy="most_frequent")
                    imputer.fit(df)
                    df = pd.DataFrame(imputer.transform(df), columns=df.columns)
                    imputerStatus = False
        else:
            for attribute in df.columns:
                if isinstance(attribute, str):
                    st.error("There are variables which cannot be used for the analysis")
                    flag = False
                    break
        if disease == 'parkinsons_udprs':
            labels = [0, 1]
            df['severity'] = pd.cut(df[targetVariable], bins=bins, labels=labels)
            y = df['severity']
            df = df.drop([targetVariable, variables,'severity'], axis=1)
        else:
            y = df[targetVariable]
            df = df.drop(targetVariable, axis=1)
        if show_info and disease == 'parkinsons_udprs' and flag:
            st.write(df.head())
        if flag:
            x = df
            if labelEncoderStatus:
                st.error("Label Encoder is required")
            elif imputerStatus:
                st.error("Imputer is required")
            elif requirement == 'Prediction':
                prediction(df, disease, col_indexs, variable, col_removed, show_info)
            elif requirement == 'Analysis Report':
                analysisReport(disease, x, y)
    # else:
    #     st.error("Please enter valid input")



def prediction(df, disease, col_indexs, ignore_variables, col_removed, show_info):
    df['result'] = 'Not diseased'
    for index, row in df.iterrows():
        if disease == "parkinsons":
            result = detection.parkinsons_prediction(list(row[:-1]))
        elif disease == 'chronic':
            result = detection.chronic_detection(list(row[:-1]))
        elif disease == 'parkinsons_udprs':
            result = detection.parkinsonsUpdrs_prediction(list(row[:-1]))
            if result[0] == 0:
                df.at[index, 'result'] = 'Non - Severe'
            else:
                df.at[index, 'result'] = 'Severe'
        if disease != 'parkinsons_udprs':
            if result[0] == 0:
                df.at[index, 'result'] = 'Not diseased'
            else:
                df.at[index, 'result'] = 'Diseased'
    df.insert(col_indexs, ignore_variables, col_removed)
    if show_info:
        st.write(df.head())
    df.to_csv('updated_data.csv', index=False)
    st.download_button(
        label='Download file',
        data=df.to_csv().encode('utf-8'),
        file_name='updated_data.csv',
        mime='text/csv'
    )
    

def analysisReport(disease, x, y):
    st.header("K - fold Cross Validation Classification")
    splits = st.text_input("Number of splits",placeholder="Number of splits", value=2)
    if not splits:
        st.error("Please enter a value.")
    elif int(splits) < 2:
        st.error("Number of splits should be greater than or equal to 2")
    elif splits.isdigit() and int(splits) <= 10:
        xTrain, xTest, yTrain, yTest = detection.splittingDataset(x, y)
        rf_scores, svm_scores, xgb_scores = detection.applyKFold(splits, x, y)
        selected_options = st.multiselect("Select an algorithm", ["Support Vector Machine", "Random Forest","XGradient Boost"], default=None, help="Please select options...", key="my_multiselect")
        if "Support Vector Machine" in selected_options:
            scores, mean = detection.kFoldResult(svm_scores)
            displayKFoldResult("Support Vector Machine", scores, mean)

        if "Random Forest" in selected_options:
            scores, mean = detection.kFoldResult(rf_scores)
            displayKFoldResult("Random Forest Classifier", scores, mean)

        if "XGradient Boost" in selected_options:
            scores, mean = detection.kFoldResult(xgb_scores)
            displayKFoldResult("XGBoost Classifier", scores, mean)

    elif int(splits) > 10:
        st.error("Number of splits should be less than 10")
    else:
        st.error("Please enter the intger")
    st.header("Applying a Model")
    
    selected_option = st.selectbox("Select an algorithm", ["Support Vector Machine", "Random Forest","XGradient Boost"], help="Please select options...")
    if selected_option == 'Support Vector Machine':
        svm_model = detection.getmodelClassification("SVM")
        trainPreds, testPreds, trainAccuracy, testAccuracy = detection.applyModel(svm_model, x, y)
        displayModelResult(trainAccuracy, testAccuracy, yTrain, yTest, trainPreds, testPreds, "Support Vector Machine")

    if selected_option == 'Random Forest':
        rf_model = detection.getmodelClassification("Random Forest")
        trainPreds, testPreds, trainAccuracy, testAccuracy = detection.applyModel(rf_model, x, y)
        displayModelResult(trainAccuracy, testAccuracy, yTrain, yTest, trainPreds, testPreds, "Random Forest Classifier")
    
    if selected_option == 'XGradient Boost':
        xgb_model = detection.getmodelClassification("XGBoost")
        trainPreds, testPreds, trainAccuracy, testAccuracy = detection.applyModel(xgb_model, x, y)
        displayModelResult(trainAccuracy, testAccuracy, yTrain, yTest, trainPreds, testPreds, "XGBoost Classifier")