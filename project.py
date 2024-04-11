try:

    import streamlit as st
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    
    st.title("CIS 335 Project By: Ayden Bixler and Matthew Janatello")

    st.write("Please download the Titanic-Dataset.csv file from: www.kaggle.com/datasets/yasserh/titanic-dataset")
    
    uploadedfile = st.file_uploader("Please insert the Titanic-Dataset.csv file")

    df = pd.read_csv(uploadedfile)

    df = df.drop(['PassengerId', 'Embarked', 'Cabin', 'Ticket', 'Name'], axis=1)

    for ele in df:
        df.dropna(axis=0,inplace=True)

    st.write(df)
    
    train, test = train_test_split(df, test_size=0.2)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    xtrain = train[features]
    ytrain = train['Survived']
    xtest = test[features]
    ytest = test['Survived']
    
    with st.sidebar:
        normalization = st.radio(
            "What Normalization Technique would you like to use?",
            ("No Normalization", "Min Max Normalization", "Z-Score")
    )
    with st.sidebar:
        selected_classifier = st.selectbox(
            "What Classification Method would you like to use?",
            ("Decision Tree", "SVM", "Adaboost", "Random Forest")
    )

    if normalization == "Min Max Normalization":
        xtrain_normalized = MinMaxScaler().fit_transform(xtrain)
        xtest_normalized = MinMaxScaler().fit_transform(xtest)
    elif normalization == "Z-Score":
        xtrain_normalized = StandardScaler().fit_transform(xtrain)
        xtest_normalized = StandardScaler().fit_transform(xtest)
    else:
        xtrain_normalized = xtrain
        xtest_normalized = xtest

    if selected_classifier == "Min Max Normalization":
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold()),
        ('classifier', selected_classifier)
        ])
    elif selected_classifier == "Z-Score":
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold()),
        ('classifier', selected_classifier)
        ])
    elif selected_classifier == "SVM":
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold()),
        ('classifier', selected_classifier)
        ])
    else:
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold()),
        ('classifier', selected_classifier)
        ])

    st.write(xtrain)
    st.write(xtest)
    
    pipe.fit(xtrain, ytrain)

except ValueError as ve:
    print("")

