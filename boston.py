import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')
st.write('**Description of Dataset**')
st.write('**CRIM** - per capita crime rate by town')
st.write('**ZN** - proportion of residential land zoned for lots over 25,000 sq.ft.')
st.write('**INDUS** - proportion of industry occupied acres per town.')
st.write('**CHAS** - Presence of River')
st.write('**NOX** - Nitrogen Oxide Concentration (Pollutant)')
st.write('**RM** -  average number of rooms per dwelling')
st.write('**AGE** - proportion of owner-occupied units')
st.write('**DIS** - distance from office')
st.write('**RAD** - distance from highway')
st.write('**TAX** - property tax')
st.write('**PTRATIO** - student to teacher ratio')
st.write('**B** - perrcentage of lower status of the population')
st.write('**LSTAT** - per capita crime rate by town')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

df_boston = pd.DataFrame(boston.data,columns=boston.feature_names)
df_boston['target'] = pd.Series(boston.target)
st.write(df_boston)

#Visualisation
chart_select = st.sidebar.selectbox(
    label ="Type of chart",
    options=['Scatterplots','Lineplots','Histogram','Boxplot']
)

numeric_columns = list(df_boston.select_dtypes(['float','int']).columns)

if chart_select == 'Scatterplots':
    st.sidebar.subheader('Scatterplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.scatter(data_frame=df_boston,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Histogram':
    st.sidebar.subheader('Histogram Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        plot = px.histogram(data_frame=df_boston,x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Lineplots':
    st.sidebar.subheader('Lineplots Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.line(df_boston,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Boxplot':
    st.sidebar.subheader('Boxplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.box(df_boston,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM',float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of Median Value of House (MEDV)')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
if st.button('Show SHAP Graphs'):
    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')
    st.write('---')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
