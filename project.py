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
    
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    xtrain = dftrain[features]
    ytrain = dftrain['Survived']
    xtest = dftest[features]
    ytest = dftest['Survived']
    
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
        x_normalized = MinMaxScaler().fit_transform(x)
    elif normalization == "Z-Score":
        x_normalized = StandardScaler().fit_transform(x)
    else:
        x_normalized = x

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

    pipe.fit(xtrain, ytrain)

except ValueError as ve:
    print("")

