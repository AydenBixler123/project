

import streamlit as st
import pandas as pd

st.title("CIS 335 Project By: Ayden Bixler and Matthew Janatello")

uploadedfile = st.file_uploader("Please insert the Titanic-Dataset.csv file")

df = pd.read_csv(uploadedfile)

df = df.drop(['PassengerId', 'Embarked', 'Cabin', 'Ticket', 'Name'], axis=1)

for ele in df:
    df.dropna(axis=0,inplace=True)

st.write(df)


