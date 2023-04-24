import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import io
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor
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

def getModelRegerssion(model):

    svm_regressor = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
    xgb_regressor = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3))
    rf_regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))

    models = {

        "SVM" : svm_regressor,
        "Random Forest" : rf_regressor,
        "XGBoost" : xgb_regressor
    }
    return models[model]

def parkinsons_prediction(input_data):
    parkinsons_model = pickle.load(open('C:/Users/karti/Disease Detection/models/trained_model_parkinsons.sav', 'rb'))
    
    input_data = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data.reshape(1, -1)

    return parkinsons_model.predict(input_data_reshaped)

def chronic_detection(input_data):

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

def applyKFoldRegression(model, x, y, splits):

    mse_scores = -cross_val_score(getModelRegerssion(model), x, y, cv=int(splits), scoring='neg_mean_squared_error')
    mae_scores = -cross_val_score(getModelRegerssion(model), x, y, cv=int(splits), scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(getModelRegerssion(model), x, y, cv=int(splits), scoring='r2')
    kFoldResultRegression(mse_scores, mae_scores, r2_scores, model)

def kFoldResultRegression(mse_scores, mae_scores, r2_scores, model):
    st.markdown("The Mean Squared Error scores for "+model+":")
    st.write(mse_scores.reshape(1, -1))
    st.write("The Root mean squared error(RMSE) is: ")
    st.write(+np.sqrt(np.mean(mse_scores)))
    st.markdown("The Mean Absolute Error scores for "+model+":")
    st.write(mae_scores.reshape(1, -1))
    st.markdown("The mean of the Mean absolute error is: ")
    st.write(np.mean(mae_scores))
    st.markdown("The Co - efficient of Determination(R2) scores for "+model+":")
    st.write(r2_scores.reshape(1, -1))
    st.markdown("The mean of R2 - scores is: ")
    st.write(np.mean(r2_scores))
    
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
    report = classification_report(yTest, yPred, digits=2)
    for i in range(10):
        report = report.replace(f'{i}.00\t', f'{i}%\t').replace(f'{i}.00\n', f'{i}%\n')
    for i in range(1, 10):
        report = report.replace(f'{i}0.0\t', f'{i}0%\t').replace(f'{i}0.0\n', f'{i}0%\n')
    report = report.replace('100.0\t', '100%\t').replace('100.0\n', '100%\n')
    return report

def applyModelRegression(model, x, y):
    regressionModel = getModelRegerssion(model)
    xTrain, xTest, yTrain, yTest = splittingDataset(x, y)
    regressionModel.fit(xTrain, yTrain)
    preds_test = regressionModel.predict(xTest)

    svm_mse = mean_squared_error(yTest, preds_test)
    svm_mae = mean_absolute_error(yTest, preds_test)
    svm_r2 = r2_score(yTest, preds_test)
    svm_rmse = np.sqrt(svm_mse)

    st.write("The Mean Squared Error for the "+model+" :")
    st.write(svm_mse)
    st.write("The Root Mean Squared Error for the "+model+" :")
    st.write(svm_rmse)
    st.write("The Mean Absolute Error for the "+model+" :")
    st.write(svm_mae)
    st.write("The Co - efficient of Determination(R2) for the "+model+" :")
    st.write(svm_r2)
    barChart([svm_rmse, svm_mae, svm_r2], model)

def barChart(values, model):
    metrics = ['RMSE', 'MAE', 'R2']
    fig = plt.figure(figsize=(6,4))
    bar_width = 0.5
    plt.bar(np.arange(len(metrics)), values, bar_width)
    plt.xticks(np.arange(len(metrics)), metrics)
    plt.title('Evaluation Metrics for '+model+' Regression Model')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    st.image(img)