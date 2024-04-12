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
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier
	
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
    
    normalization = st.sidebar.radio(
            "What Normalization Technique would you like to use?",
            ("No Normalization", "Min Max Normalization", "Z-Score")
    )
    selected_classifier = st.sidebar.selectbox(
            "What Classification Method would you like to use?",
            ("Decision Tree", "SVM", "Adaboost", "Random Forest")  
    )
    if "Decision Tree" in selected_classifier:
     min_samples_split = st.sidebar.slider('Choose a value for the min samples split parameter.', 1, 10, 2)
     max_depth = st.sidebar.slider('Choose a value for the max depth parameter.', 1, 5, 1)
     max_features = st.sidebar.slider('Choose a value for the max features parameter.', 1, 7, 7)
    elif "SVM" in selected_classifier:
     parameter = st.sidebar.slider('Choose a value for the  parameter.', 1, 2, 1)
    elif "Adaboost" in selected_classifier:
     parameter = st.sidebar.slider('Choose a value for the  parameter.', 1, 2, 1)
    else:
     n_estimators = st.sidebar.slider('Choose a value for the n estimators parameter.', 1, 15, 1)

	
    if normalization == "Min Max Normalization":
        xtrain_normalized = MinMaxScaler().fit_transform(xtrain)
        xtest_normalized = MinMaxScaler().fit_transform(xtest)
    elif normalization == "Z-Score":
        xtrain_normalized = StandardScaler().fit_transform(xtrain)
        xtest_normalized = StandardScaler().fit_transform(xtest)
    else:
        xtrain_normalized = xtrain
        xtest_normalized = xtest


    ##Decision Tree
    if "Decision Tree" in selected_classifier:
     pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold()),
        ('classifier', DecisionTreeClassifier())
        ])
	    
     pipe.fit(xtrain_normalized, ytrain)
	    
     parameters = {
      	'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
	'selector__threshold': [0, 0.001, 0.01],
	'classifier__n_estimators': min_samples_split,
      	'classifier__max_depth': max_depth,
	'classifier__max_features': max_features
    	}
     grid = GridSearchCV(pipe, parameters, cv=2).fit(xtrain_normalized, ytrain)

	
    #SVM
    #elif "SVM" in selected_classifier:
     #pipe = Pipeline([
        #('scaler', StandardScaler()),
        #('selector', VarianceThreshold()),
        #('classifier', SVC())
        #])

     #pipe.fit(xtrain_normalized, ytrain)
    
     #parameters = {
      	#'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
	#'selector__threshold': [0, 0.001, 0.01],
	#'classifier__': ,
      	#'classifier__': ,
        #'classifier__': 
    	#}

	
    #Adaboost
    #elif "Adaboost" in selected_classifier:
     #pipe = Pipeline([
        #('scaler', StandardScaler()),
        #('selector', VarianceThreshold()),
        #('classifier', AdaBoostClassifier())
        #])

     #pipe.fit(xtrain_normalized, ytrain)
    
     #parameters = {
      	#'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
	#'selector__threshold': [0, 0.001, 0.01],
	#'classifier__': ,
      	#'classifier__': ,
        #'classifier__': 
    	#}

	
    #RandomForest
    else:
     #pipe = Pipeline([
        #('scaler', StandardScaler()),
        #('selector', VarianceThreshold()),
        #('classifier', AdaBoostClassifier())
        #])

     #pipe.fit(xtrain_normalized, ytrain)
    
     #parameters = {
      	#'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
	#'selector__threshold': [0, 0.001, 0.01],
	#'classifier__': ,
      	#'classifier__': ,
        #'classifier__': 
    	#}


   with st.form('form1'):
     st.write('this is a form')
     st.form_submit_button('Press to Calculate')

except ValueError as ve:
    print("")

