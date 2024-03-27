try:

    import streamlit as st
    import pandas as pd
    #from sklearn.model_selection import train_test_split
    #from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.feature_selection import VarianceThreshold
    #from sklearn.pipeline import Pipeline
    #from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
    #from sklearn.ensemble import RandomForestClassifier
    #from sklearn.ensemble import BaggingClassifier
    #from sklearn.tree import DecisionTreeClassifier
    
    st.title("CIS 335 Project By: Ayden Bixler and Matthew Janatello")

    st.write("Please download the Titanic-Dataset.csv file from: www.kaggle.com/datasets/yasserh/titanic-dataset") 
    uploadedfile = st.file_uploader("Please insert the Titanic-Dataset.csv file")

    df = pd.read_csv(uploadedfile)

    df = df.drop(['PassengerId', 'Embarked', 'Cabin', 'Ticket', 'Name'], axis=1)

    for ele in df:
        df.dropna(axis=0,inplace=True)

    st.write(df)

    

except ValueError as ve:
    print("")
