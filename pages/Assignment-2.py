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
import plotly.express as px
import streamlit as st
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

# Function to create Snowflake Session to connect to Snowflake
def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("Assignment-2\connection.json"))).create()
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
    df.columns=['transaction_id', 'product_id', 'customer_id', 'transaction_date',
       'online_order', 'order_status', 'brand', 'product_line',
       'product_class', 'product_size', 'list_price', 'standard_cost',
       'product_first_sold_date']
    df.transaction_date  = pd.to_datetime(df['transaction_date'])
    df.order_status=df['order_status'].astype('str')
    # df.rename(columns={'CUSTOMER_ID':'customer_id','TRANSACTION_DATE':'transaction_date','ONLINE_ORDER':'online_order','ORDER_STATUS':'order_status'},inplace=True)
    df.transaction_date  = pd.to_datetime(df['transaction_date'])
    df_final = df[['customer_id','transaction_date','online_order','order_status']]
    df_final = df_final[df_final['order_status'] == 'True']
    df_final = df_final[~df_final.duplicated()]
    df_final['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df_final['transaction_month'] = df['transaction_date'].apply(get_month)
    #Create Cohort Month per Rows 

    group = df_final.groupby('customer_id')['transaction_month']
    df_final['cohort_month'] = group.transform('min')

    # st.write(df_final)
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

    fig = px.imshow(retention, text_auto=True)
    st.plotly_chart(fig)

# Streamlit config
st.set_page_config("Assignment-2: Cohort Analysis", "centered")
st.write("<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>", unsafe_allow_html=True)
st.title("Assignment-2: Cohort Analysis")

# Call functions to get Snowflake session and load data
session = create_session()
transactions=load_data()

st.write(transactions)
cohortAnalysis(transactions)
