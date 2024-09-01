import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import plotly.express as px
import shap

df=pd.read_csv("weatherAUS.csv")

#df_final=pd.read_csv(".\Final\weatherAUS_Agglomeration.csv")

#target = df_final['RainTomorrow']
#data = df_final.drop(['RainTomorrow', 'Date', 'Location'], axis = 1)
#X, X_valid, y, y_valid = train_test_split(data, target, test_size=0.1, random_state=10)
#rOs = RandomOverSampler()
#X_ro, y_ro = rOs.fit_resample(X, y)
#X_train_ro, X_test_ro, y_train_ro, y_test_ro = train_test_split(X_ro, y_ro, test_size=0.2, random_state=20)
#train_ro = xgb.DMatrix(data=X_train_ro, label=y_train_ro)
#test_ro = xgb.DMatrix(data=X_test_ro, label=y_test_ro)
#valid = xgb.DMatrix(data=X_valid, label=y_valid)

xgb_model = joblib.load('xgb_model.sav')

st.title("Rain Prediction in Australia")
st.sidebar.title("Table of contents")
pages=["Data Exploration", "Data Vizualization", "Preprocessing", "Modelling", "Interpretability"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
    st.write("## Data Exploration") 

    st.markdown('''We first take a look at the standard analysis of the data set.  
             For this we can choose the aspect of the data that interests us:''')
    display_0 = st.radio('Data Aspect:', ('Shape', 'Examples', 'Description', 'Missing Values'))

    if display_0 == 'Shape':
        st.write('The shape of the data set is:', df.shape)
    elif display_0 == 'Examples':
        st.dataframe(df.head(10))
    elif display_0 == 'Description':
        st.dataframe(df.describe())
    elif display_0 == 'Missing Values':
        st.dataframe(df.isna().sum())


if page == pages[1] : 
    st.write("## Data Vizualization")

    st.markdown('''Of particular interest are the distribution of the target variable RainTomorrow,
                a Heatmap to study the correlation between the variables 
                and the distribution of certain features to detect outliers.''')

    choice_0 = ['Target Distribution', 'Correlation Heatmap', 'Feature Distribution']
    option_0 = st.selectbox('Feature to vizualize:', choice_0)
    st.write('The', option_0, 'is shown.')

    if option_0 == 'Target Distribution':
        #plot of rainTomorrow imbalance
        fig = plt.figure(figsize=(16, 8))
        sns.countplot(x = 'RainTomorrow', data = df)
        plt.title('Distribution of Target Variable RainTomorrow')
        plt.xlabel('RainTomorrow')
        st.pyplot(fig)
    elif option_0 == 'Correlation Heatmap':
        #plot heatmap for correlation overview
        data_quant = df.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'], axis = 1)
        fig = plt.figure(figsize=(13,13))
        sns.heatmap(data_quant.corr(),  annot=True, cmap="RdBu_r", center =0)
        st.pyplot(fig)
    elif option_0 == 'Feature Distribution':
        #plot of Evaporation distribution
        fig = plt.figure(figsize=(16, 8))
        sns.boxplot(data=df[['Evaporation']])
        plt.title('Box Plot of Evaporation')
        plt.ylabel('A pan evaporation in mm')
        st.pyplot(fig)

        #plot of Min and Max Temp distribution
        fig = plt.figure(figsize=(16, 8))
        sns.boxplot(data=df[['MinTemp', 'MaxTemp']])
        plt.title('Box Plot of Min and Max Temp')
        plt.ylabel('Temperature in \u2103')
        st.pyplot(fig)

        #plot of Rainfall distribution
        fig = plt.figure(figsize=(16, 8))
        sns.boxplot(data=df[['Rainfall']])
        plt.title('Box Plot of Rainfall')
        plt.ylabel('Height in mm')
        st.pyplot(fig)


if page == pages[2] : 
    st.write("## Preprocessing")

    winddir = {'Direction': ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'],
               'Dir_x': ["cos(0)", "cos(π/8)", 'cos(2π/8)', 'cos(3π/8)',
                          'cos(4π/8)', 'cos(5π/8)', 'cos(6π/8)'
                          , 'cos(7π/8)', 'cos(π)', 'cos(9π/8)', 
                          'cos(10π/8)', 'cos(11π/8)', 'cos(12π/8)'
                          , 'cos(13π/8)', 'cos(14π/8)', 'cos(15π/8)'],
                'Dir_y': ["sin(0)", "sin(π/8)", 'sin(2π/8)', 'sin(3π/8)',
                          'sin(4π/8)', 'sin(5π/8)', 'sin(6π/8)'
                          , 'sin(7π/8)', 'sin(π)', 'sin(9π/8)', 
                          'sin(10π/8)', 'sin(11π/8)', 'sin(12π/8)'
                          , 'sin(13π/8)', 'sin(14π/8)', 'sin(15π/8)']
                          }
    df_wind = pd.DataFrame(data=winddir) 

    monthdir = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
               'Month_x': ["cos(0)", "cos(π/6)", 'cos(2π/6)', 'cos(3π/6)',
                          'cos(4π/6)', 'cos(5π/6)', 'cos(π)'
                          , 'cos(7π/6)', 'cos(8π/6)', 
                          'cos(9π/6)', 'cos(10π/6)', 'cos(11π/6)'],
                'Month_y': ["sin(0)", "sin(π/6)", 'sin(2π/6)', 'sin(3π/6)',
                          'sin(4π/6)', 'sin(5π/6)', 'sin(π)'
                          , 'sin(7π/6)', 'sin(8π/6)', 
                          'sin(9π/6)', 'sin(10π/6)', 'sin(11π/6)']
                          }
    df_month = pd.DataFrame(data=monthdir)

    binary= {'Boolean': ['Yes', 'No', 'NaN'], 'Numerical': [1, 0, 'NaN']}
    df_bin = pd.DataFrame(data=binary)

    df_cluster = df_final[['Location', 'Cluster']].drop_duplicates().sort_values(by=['Cluster'])

    st.markdown('''In order to optimally train the modell some features had to be processed.  
                The way this was done depended on the type and distribution of the variable.''')
  
    display = st.radio('Preprocessed feature', ('Wind Directions', 'Month', 'Binary Variables', 'Highest Evaporation', 'Missing Target', 'Other Missing Values', 'Cluster'))
    if display == 'Wind Directions':
        st.dataframe(df_wind.set_index(df_wind.columns[0]))
    elif display == 'Month':
        st.write(df_month.set_index(df_month.columns[0])) 
    elif display == 'Binary Variables':
        st.write(df_bin.set_index(df_bin.columns[0]))
    elif display == 'Highest Evaporation':
        st.write('Before the value 145 in row 42358 there are 18 NaN, they were replaced by the mean:', round(145/19,1))
        st.write('Before the value 86.2 in row 8831 there are 11 NaN, they were replaced by the mean:', round(86.2/12, 1))
        st.write('Before the value 82.4 in row 106968 there are 3 NaN, they were replaced by the mean:', round(82.4/4, 1))
        st.write('Before the value 81.2 in row 105935 there are 4 NaN, they were replaced by the mean:', round(81.2/5, 1))
    elif display == 'Missing Target':
        st.write('Rows with missing target values were deleted.')
    elif display == 'Other Missing Values':
        st.write('Other missing values were replaced using a KNN-Imputation with n_neighboors = 5.')
    elif display == 'Cluster':
        st.write('Using an agglomerative clustering the locations were clustered into 5 cluster.')
        st.dataframe(df_cluster.set_index(df_cluster.columns[0])
)

if page == pages[3] : 
    st.write("## Modelling")
    
    st.write("### XGB Model")
    def prediction(dataset):
        if dataset == 'Training':
            preds = xgb_model.predict(train_ro)
            target = y_train_ro
        elif dataset == 'Test':
            preds = xgb_model.predict(test_ro)
            target = y_test_ro
        elif dataset == 'Validation':
            preds = xgb_model.predict(valid)
            target = y_valid
        xgbpreds = pd.Series(np.where(preds > 0.6, 1, 0))
        return [xgbpreds, target]
    
    st.markdown('''The best results for predicting the target variable RainTomorrow were obtained from an XGB Model.  
                For Training and Evaluating the model the Data Set was split into a Training, Test and Validation set.  
                The Test set was used for testing during the training, while the validation set was never seen by the modell.  
                Below are the scores of the best modell for each data set.''')

    choice = ['Training', 'Test', 'Validation']
    option = st.selectbox('Choice of the data', choice)
    st.write('The scores are displayed for the ', option, 'Data')

    report = pd.DataFrame(classification_report(pd.Series(prediction(option)[1]).reset_index(drop=True), prediction(option)[0], output_dict=True)).transpose()
    crosstab = pd.crosstab(prediction(option)[0], pd.Series(prediction(option)[1]).reset_index(drop=True), colnames = ['Predictions'], rownames = ['Observations'])

    display = st.radio('What do you want to show?', ('Classification report', 'Confusion matrix'))
    if display == 'Classification report':
        st.dataframe(report)
    elif display == 'Confusion matrix':
        st.write(crosstab)

    st.write("### SKTime Models")

    st.markdown('''For a time series prediction threee different Forecasters were trained on each of the different clusters of locations.  
                The results can be seen for each of the clusters and Forecasters.''')

    choice_2 = ['0', '1', '2', '3', '4']
    option_2 = st.selectbox('Choice of cluster', choice_2)
    st.write('The forecast is displayed for the cluster', option_2)
    
    display_2 = st.radio('Which type of forecaster shall be shown?', ('Naive Forecaster', 'ExpandingWindowSplitter', 'AutoARIMA'))
    if display_2 == 'Naive Forecaster':
        image = "{model}_{clus}.png".format(model="naive", clus=option_2)
        st.image(image)
    elif display_2 == 'ExpandingWindowSplitter':
        image = "{model}_{clus}.png".format(model="ews", clus=option_2)
        st.image(image)
    elif display_2 == 'AutoARIMA':
        image = "{model}_{clus}.png".format(model="arima", clus=option_2)
        st.image(image)
        

if page == pages[4] : 
    st.write("## Interpretability")

    shap_values = joblib.load('shap_values_model')
    shap_sample = shap_values.sample(100, random_state=10) 
    explainer = shap.Explainer(xgb_model)

    st.markdown('''Using SHAP it is possible to gain insights into the importance of different features in the XGB model.  
                In the first graph any individuell entry can be analyzed.  
                A Decision Plot for a sample of 100 entries can be shown to see typical structures of the decision.  
                Finally the Summary Plot allows to see the importance of each feature on the modell as a whole.''')

    entry = st.number_input("Enter the number of the entry that shall be analyzed:", value = 1)
    fig = plt.figure(figsize=(16, 8))    
    shap.plots.waterfall(shap_values[entry])
    st.pyplot(fig)

    if st.checkbox("Show Decision Plot"):
        fig = plt.figure(figsize=(16, 8))  
        shap.decision_plot(explainer.expected_value, shap_sample.values, X_train_ro)
        st.pyplot(fig)

    if st.checkbox("Show Summary Plot"):
        fig = plt.figure(figsize=(16, 8))  
        shap.summary_plot(shap_values)
        st.pyplot(fig)
