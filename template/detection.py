import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import io
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import streamlit as st

def getmodelClassification(model):

    rf_model = RandomForestClassifier(random_state=42)
    svm_model = SVC(random_state=42, probability=True)
    xgb_model = XGBClassifier()
    models = {
        "SVM" : svm_model,
        "Random Forest" : rf_model,
        "XGBoost" : xgb_model
    }
    return models[model]

def parkinsons_prediction(input_data):
    try:
        parkinsons_model = pickle.load(open('models/trained_model_parkinsons.sav', 'rb'))
    except:
        parkinsons_model = pickle.load(open('C:/Users/karti/Disease Detection/models/trained_model_parkinsons.sav', 'rb'))

    input_data = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data.reshape(1, -1)

    return parkinsons_model.predict(input_data_reshaped)

def parkinsonsUpdrs_prediction(input_data):
    try:
        parkinsons_model = pickle.load(open('models/trained_model_parkinsons_udprs.sav', 'rb'))
    except:
        parkinsons_model = pickle.load(open('C:/Users/karti/Disease Detection/models/trained_model_parkinsons_udprs.sav', 'rb'))

    input_data = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data.reshape(1, -1)

    return parkinsons_model.predict(input_data_reshaped)

def chronic_detection(input_data):
    try:
        chronic_model = pickle.load(open('models/trained_model_chronic_kidney.sav','rb'))
    except:
        chronic_model = pickle.load(open('C:/Users/karti/Disease Detection/models/trained_model_chronic_kidney.sav','rb'))

    input_data = np.asarray(input_data, dtype=np.float64)

    input_data_reshaped = input_data.reshape(1, -1)
    
    return chronic_model.predict(input_data_reshaped)

def splittingDataset(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

def applyKFold(splits, x, y):

    kfold = KFold(n_splits=int(splits), shuffle=True, random_state=42)

    rf_scores = cross_val_score(getmodelClassification("Random Forest"), x, y, cv=kfold)
    svm_scores = cross_val_score(getmodelClassification("SVM"), x, y, cv=kfold)
    xgb_scores = cross_val_score(getmodelClassification("XGBoost"), x, y, cv=kfold)

    return rf_scores, svm_scores, xgb_scores

    
def kFoldResult(model):
    return model.reshape(1, -1), (model.mean()*100)

def applyModel(model, x, y):

    xTrain, xTest, yTrain, yTest = splittingDataset(x, y)
    
    model.fit(xTrain, yTrain)
    preds_test = model.predict(xTest)
    preds_train = model.predict(xTrain)

    return preds_train, preds_test, accuracy_score(yTrain, model.predict(xTrain))*100, accuracy_score(yTest, preds_test)*100

def displayConfusioMatrix(y, pred):
    cf_matrix = confusion_matrix(y, pred)
    fig = plt.figure(figsize=(6,4))
    sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    st.image(img)

def classificationReport(yTest, yPred):
    return classification_report(yTest, yPred, digits=2)