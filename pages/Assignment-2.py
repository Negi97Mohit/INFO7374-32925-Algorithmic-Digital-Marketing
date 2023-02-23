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
# cohortAnalysis(transactions)
