import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from pivottablejs import pivot_ui
import streamlit.components.v1 as components
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
import warnings
import itertools


# Define all data
input = pd.read_csv('./tsInput.csv')
ref1 = pd.read_csv('./project_Ls.csv')
ref2 = pd.read_csv('./staff_Lvl_Gp.csv')
ref3 = pd.read_csv('./salary_Lvl.csv')
data = './data.csv'

# Merging master-input file with reference-file
input_ref1 = pd.merge(input, ref1, on='PROJECT', how='left')
input_ref1_2 = pd.merge(input_ref1, ref2, on='STAFF', how='left')
input_ref1_2_3 = pd.merge(input_ref1_2, ref3, on='LEVEL', how='left')

# Adding new calculated column calculated to the merge-table
input_ref1_2_3['SalaryCost'] = input_ref1_2_3['HOUR'] * input_ref1_2_3['AVER HOURLY SALARY']

# Output the .csv data file to a specific location
outputTable = pd.DataFrame(input_ref1_2_3)
outputTable.to_csv(data)

# Load the processed output data into streamlit as data source
DATA_URL = ("./data.csv")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
data = load_data()


# Setting up streamlit by giving titles and description
st.write(
    '###### REMARK: The analysis on this web app is based on mock datasets. This site is used by the owner as a means to practice and illustrate skills in python for data analysis and visualization. To access the control sidebar, click on the arrow icon on the top left corner.')
st.title('Project labour costs')
st.subheader('Scenario')
st.write(
    'A pharmaceutical research institute aims to analyze its labor costs associated with various research projects. To achieve this, a timesheet system was developed to collect the time spent by each employee. The data is grouped by functional groups and projects, allowing for insights into labor expenditures and resource allocation.')
st.markdown('Employee number: 64')
st.markdown('Functional group: 7')
st.markdown('Project number: 17')
st.markdown('Data collection period: 1st Jan 2023 - 14th Sep 2024')
st.subheader('View of data for selected month')

st.sidebar.title('Interactive control sidebar')
st.sidebar.subheader('View of data for selected month')


# Graph 1: labour costs in S$ and Hr by Project/Functional Group
selectYr = st.sidebar.selectbox('Year', [2023, 2024], key='1')
if selectYr == int('2023'):
    selectMth = st.sidebar.selectbox('Month', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], key='2')
else:
    selectMth = st.sidebar.selectbox('Month', [1, 2, 3, 4, 5, 6, 7, 8, 9], key='3')

selectGraph = data.query('(YEAR == @selectYr) & (MONTH == @selectMth)')
select = st.sidebar.selectbox('Sort by:', ['Functional Group', 'Project'], key='4')
if select == 'Functional Group':
    trace0 = go.Bar(x=selectGraph["GROUP"], y=selectGraph["SalaryCost"], name='S$', xaxis='x', yaxis='y', offsetgroup=1)
    trace1 = go.Bar(x=selectGraph["GROUP"], y=selectGraph["HOUR"], name='Hr', yaxis='y2', offsetgroup=2)
    dataTrace = [trace0, trace1]
    layoutTrace = {
        'xaxis': {'title': 'Functional Group'},
        'yaxis': {'title': 'Salary Cost (S$)'},
        'yaxis2': {'title': 'Time Spent (Hr)', 'overlaying': 'y', 'side': 'right'},
        'height': 600,  # **Increased height for better spacing**
        'legend': {
            'x': 1.1,          # **Further moved legend away from the graph**
            'y': 1,            # **Positioned legend at the top**
            'xanchor': 'left',
            'yanchor': 'top',
            'orientation': 'v'  # **Vertical orientation**
        },
        'margin': {'r': 120}  # **Further increased right margin for y-axis2 label space**
    }
    fig = go.Figure(data=dataTrace, layout=layoutTrace)
    st.plotly_chart(fig)

    # Graph 2b: Ranking of labour cost for project/Functional Group by S$ & Hr
    selectGraph2 = px.bar(selectGraph, x='SalaryCost', y='GROUP', color='PROJECT', facet_row=None, category_orders={},
                          labels={})
    selectGraph2.update_yaxes(categoryorder='sum ascending')
    st.plotly_chart(selectGraph2)

    selectGraph2 = px.bar(selectGraph, x='HOUR', y='GROUP', color='PROJECT', facet_row=None, category_orders={},
                          labels={})
    selectGraph2.update_yaxes(categoryorder='sum ascending')
    st.plotly_chart(selectGraph2)
else:
    trace0 = go.Bar(x=[selectGraph["FIELD"], selectGraph["PROJECT"]], y=selectGraph["SalaryCost"], name='S$', xaxis='x',
                    yaxis='y', offsetgroup=1)
    trace1 = go.Bar(x=[selectGraph["FIELD"], selectGraph["PROJECT"]], y=selectGraph["HOUR"], name='Hr', yaxis='y2',
                    offsetgroup=2)
    dataTrace = [trace0, trace1]
    layoutTrace = {'xaxis': {'title': 'Project'}, 'yaxis': {'title': 'Salary Cost (S$)'},
                   'yaxis2': {'title': 'Time Spent (Hr)', 'overlaying': 'y', 'side': 'right'}, 'height': 600,
                   'legend': {
                        'x': 1.1,            # **Positioning the legend outside the graph on the right**
                        'y': 1,               # **Set y position to be at the top of the graph**
                        'xanchor': 'left',    # Anchor to the left
                        'yanchor': 'top',     # Anchor to the top
                        'orientation': 'v'     # **Vertical orientation**
                    },
                    'margin': {'r': 120}    # **Increase right margin to create space for y-axis2 label**
                   }
    fig = go.Figure(data=dataTrace, layout=layoutTrace)
    st.plotly_chart(fig)

    # Graph 2a: Ranking of labour cost for project/Functional Group by S$ & Hr
    selectGraph2 = px.bar(selectGraph, x='SalaryCost', y='PROJECT', color='GROUP', facet_row=None, category_orders={},
                          labels={})
    selectGraph2.update_yaxes(categoryorder='sum ascending')
    st.plotly_chart(selectGraph2)

    selectGraph2 = px.bar(selectGraph, x='HOUR', y='PROJECT', color='GROUP', facet_row=None, category_orders={},
                          labels={})
    selectGraph2.update_yaxes(categoryorder='sum ascending')
    st.plotly_chart(selectGraph2)

# Hide/Show data table for the selected month
if st.sidebar.checkbox("Show data table", False):
    st.markdown("#### Data table for the selected time period")
    selectGraph = data.query('(YEAR == @selectYr) & (MONTH == @selectMth)')
    selectGraph = selectGraph.drop(['Unnamed: 0', 'WEEKNUMBER', 'CREATED_AT', 'LEVEL', 'AVER HOURLY SALARY'], axis=1)
    selectGraph

# Appendices: pivot table and forecasting data
st.subheader('Appendices')

# Hide/Show pivot table for raw data
st.sidebar.subheader("Appendices")
if st.sidebar.checkbox("Show pivot table for the raw data", True):
    st.markdown("#### Pivot table for the raw data")
    dataPv = data.drop(['AVER HOURLY SALARY'], axis=1)
    pvTable = pivot_ui(dataPv)
    with open(pvTable.src) as t:
        components.html(t.read(), height=400, scrolling=True)

# Below is the machine learning time series forecasting for time enteries
dataPrd = data
dataPrd = pd.pivot_table(dataPrd, index=['CREATED_AT'], values='HOUR', aggfunc='sum', margins=True)
dataPrd = dataPrd[:-1]
dataPrd = pd.DataFrame(dataPrd.to_records())

dates = list(dataPrd['CREATED_AT'])
dates = list(pd.to_datetime(dates))
Hr = list(dataPrd['HOUR'])

dataset = pd.DataFrame(columns=['ds', 'y'])
dataset['ds'] = dates
dataset['y'] = Hr
dataset = dataset.set_index('ds')

index = pd.date_range(start=dataset.index.min(), end=dataset.index.max(), freq='D')
dataset = dataset.reindex(index)
dataset = dataset.loc['2023-01-01':'2024-09-14']
dataset['y'] = dataset['y'].fillna(0)

start_date = '2023-12-30'
train = dataset.loc[dataset.index < pd.to_datetime(start_date)]
test = dataset.loc[dataset.index >= pd.to_datetime(start_date)]
model = SARIMAX(train, order=(3, 0, 7))
results = model.fit(disp=True)

sarimax_prediction = results.predict(start='2023-12-30', end='2024-09-13', dynamic=False)
sarimax_prediction = pd.DataFrame(sarimax_prediction)

trace1 = {
    "name": "Observation",
    'mode': 'lines',
    'type': 'scatter',
    'x': dataset.index,
    'y': dataset['y']
}
trace2 = {
    'name': 'Prediction',
    'mode': 'lines',
    'type': 'scatter',
    'x': sarimax_prediction.index,
    'y': sarimax_prediction['predicted_mean']
}
data = [trace1, trace2]
layout = {
    "title": 'Method: SARIMAX',
    "xaxis": {'type': 'date', "title": "Dates", 'autorange': True},
    "yaxis": {"title": "Time entered (Hr)"},
    'autosize': True
}
fig = Figure(data=data, layout=layout)

# Hide/show forecasting chart
if st.sidebar.checkbox("Show forecasting for time entries", True):
    st.markdown("#### Time series prediction using SARIMAX")
    st.plotly_chart(fig)

# Print the SARIMA MAE and model summary
st.write('SARIMA MAE = ', mean_absolute_error(sarimax_prediction, test))


# Load the forecasting model
@st.cache_resource
def load_model():
    model = pm.auto_arima(train, start_p=0, start_q=0, test='adf', max_p=7, max_q=7, m=1, d=None, seasonal=False, start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    return model
model = load_model()


st.text(model.summary())

