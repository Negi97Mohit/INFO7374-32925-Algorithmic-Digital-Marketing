# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:11:43 2023

@author: mohit
"""
import json
import altair as alt
import pandas as pd
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col
import streamlit as st
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

APP_ICON_URL = "https://i.imgur.com/dBDOHH3.png"

# Function to create Snowflake Session to connect to Snowflake
def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("connection.json"))).create()
        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session

# Function to load last six months' budget allocations and ROI 
@st.experimental_memo(show_spinner=False)
def load_data():
    transaction_data = session.table("TRANSACTIONS").to_pandas()
    return transaction_data

def get_month(x) :
    return dt.datetime(x.year, x.month,1)

#Calculate Cohort Index for Each Rows

def get_date_int(df, column) :
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    
    return year, month, day


def cohortAnalysis(df):
    df.rename(columns={'CUSTOMER_ID':'customer_id','TRANSACTION_DATE':'transaction_date','ONLINE_ORDER':'online_order','ORDER_STATUS':'order_status'},inplace=True)  
    df_final = df[['customer_id','transaction_date','online_order','order_status']]
    df_final = df_final[df_final['order_status'] == 'Approved']
    df_final = df_final[~df_final.duplicated()]
    df_final['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df_final['transaction_month'] = df['transaction_date'].apply(get_month)
    #Create Cohort Month per Rows 

    group = df_final.groupby('customer_id')['transaction_month']
    df_final['cohort_month'] = group.transform('min')


    transaction_year, transaction_month, transaction_day = get_date_int(df_final, 'transaction_month')
    cohort_year, cohort_month, cohort_day = get_date_int(df_final,'cohort_month')
    #Calculate Year Differences
    years_diff = transaction_year - cohort_year

    #Calculate Month Differences
    months_diff = transaction_month - cohort_month

    df_final['cohort_index'] = years_diff*12 + months_diff + 1
    #Final Grouping to Calculate Total Unique Users in Each Cohort
    cohort_group = df_final.groupby(['cohort_month','cohort_index'])

    cohort_data = cohort_group['customer_id'].apply(pd.Series.nunique)
    cohort_data = cohort_data.reset_index()

    cohort_counts = cohort_data.pivot_table(index = 'cohort_month',
                                            columns = 'cohort_index',
                                            values = 'customer_id'   
                                        )
    #Calculate Retention rate per Month Index

    cohort_size = cohort_counts.iloc[:,0]

    retention = cohort_counts.divide(cohort_size, axis = 0)

    retention = retention.round(3)*100

    retention.index = retention.index.strftime('%Y-%m')
    plt.figure(figsize = (16,10))

    plt.title('MoM Retention Rate for Customer Transaction Data')

    sns.heatmap(retention, annot = True, cmap="YlGnBu", fmt='g')

    plt.xlabel('Cohort Index')
    plt.ylabel('Cohort Month')
    plt.yticks(rotation = '360')

    plt.show()


# Streamlit config
st.set_page_config("Assignment-2: Cohort Analysis", APP_ICON_URL, "centered")
st.write("<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>", unsafe_allow_html=True)
st.image(APP_ICON_URL, width=80)
st.title("Assignment-2: Cohort Analysis")

# Call functions to get Snowflake session and load data
session = create_session()
transactions=load_data()

st.write(transactions)
cohortAnalysis(transactions)



# # Display advertising budget sliders and set their default values
# st.header("Advertising budgets")
# col1, _, col2 = st.columns([4, 1, 4])
# channels = ["Search engine", "Social media", "Email", "Video"]
# budgets = []
# for channel, default, col in zip(channels, df_last_months_allocations["BUDGET"].values, [col1, col1, col2, col2]):
#     with col:
#         budget = st.slider(channel, 0, 100, int(default), 5)
#         budgets.append(budget)

# # Function to call "predict_roi" UDF that uses the pre-trained model for inference
# # Note: Both the model training and UDF registration is done in Snowpark_For_Python.ipynb
# st.header("Predicted revenue")
# @st.experimental_memo(show_spinner=False)
# def predict(budgets):
#     df_predicted_roi = session.sql(f"SELECT predict_roi(array_construct({budgets[0]*1000},{budgets[1]*1000},{budgets[2]*1000},{budgets[3]*1000})) as PREDICTED_ROI").to_pandas()
#     predicted_roi, last_month_roi = df_predicted_roi["PREDICTED_ROI"].values[0] / 100000, df_last_six_months_roi["ROI"].iloc[-1]
#     change = round((predicted_roi - last_month_roi) / last_month_roi * 100, 1)
#     return predicted_roi, change

# # Call predict function upon user interaction -- i.e. everytime the sliders are changed -- to get a new predicted ROI 
# predicted_roi, change = predict(budgets)
# st.metric("", f"$ {predicted_roi:.2f} million", f"{change:.1f} % vs last month")
# months = ["January", "February", "March", "April", "May", "June", "July"]
# july = pd.DataFrame({"MONTH": ["July", "July", "July", "July"], "CHANNEL": ["SEARCHENGINE", "SOCIALMEDIA", "VIDEO", "EMAIL"], "BUDGET": budgets, "ROI": [predicted_roi] * 4})
# chart_data = pd.concat([historical_data,july]).reset_index(drop=True)
# chart_data = chart_data.replace(["SEARCHENGINE", "EMAIL", "SOCIALMEDIA", "VIDEO"], ["Search engine", "Email", "Social media", "Video"])

# # Display allocations and ROI charts
# # Note: Streamlit docs on charts can be found here: https://docs.streamlit.io/library/api-reference/charts
# base = alt.Chart(chart_data).encode(alt.X("MONTH", sort=months, title=None))
# bars = base.mark_bar().encode(
#     y=alt.Y("BUDGET", title="Budget", scale=alt.Scale(domain=[0, 400])),
#     color=alt.Color("CHANNEL", legend=alt.Legend(orient="top", title=" ")),
#     opacity=alt.condition(alt.datum.MONTH == "July", alt.value(1), alt.value(0.3)),
# )
# lines = base.mark_line(size=3).encode(
#     y=alt.Y("ROI", title="Revenue", scale=alt.Scale(domain=[0, 25])),
#     color=alt.value("#808495"),
#     tooltip=["ROI"],
# )
# points = base.mark_point(strokeWidth=3).encode(
#     y=alt.Y("ROI"),
#     stroke=alt.value("#808495"),
#     fill=alt.value("white"),
#     size=alt.condition(alt.datum.MONTH == "July", alt.value(300), alt.value(70)),
# )
# chart = alt.layer(bars, lines + points).resolve_scale(y="independent")
# chart = chart.configure_view(strokeWidth=0).configure_axisY(domain=False).configure_axis(labelColor="#808495", tickColor="#e6eaf1", gridColor="#e6eaf1", domainColor="#e6eaf1", titleFontWeight=600, titlePadding=10, labelPadding=5, labelFontSize=14).configure_range(category=["#FFE08E", "#03C0F2", "#FFAAAB", "#995EFF"])
# st.altair_chart(chart, use_container_width=True)

# # Setup the ability to save user-entered allocations and predicted value back to Snowflake 
# submitted = st.button("❄️ Save to Snowflake")
# if submitted:
#     with st.spinner("Making snowflakes..."):
#         df = pd.DataFrame({"MONTH": ["July"], "SEARCHENGINE": [budgets[0]], "SOCIALMEDIA": [budgets[1]], "VIDEO": [budgets[2]], "EMAIL": [budgets[3]], "ROI": [predicted_roi]})
#         session.write_pandas(df, "BUDGET_ALLOCATIONS_AND_ROI")
#         st.success("✅ Successfully wrote budgets & prediction to your Snowflake account!")
#         st.snow()
