try:
    import streamlit as st
    import pandas as pd
    import joblib
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
    from sklearn.model_selection import cross_val_score
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, f1_score
    
	
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
     C = st.sidebar.slider('Choose a value for the C parameter.', 1, 3, 1)
     shrinking = st.sidebar.radio("Would you like the shrinking parameter to be True/False?",
            (True, False))
     probability = st.sidebar.radio("Would you like the probability parameter to be True/False?",
            (True, False))
    elif "Adaboost" in selected_classifier: 
     n_estimators = st.sidebar.slider('Choose a value for the n estimators parameter.', 10, 100, 10, 10)
     learning_rate = st.sidebar.slider('Choose a value for the learning rate parameter.', 1, 5, 1)
    else:
     n_estimators = st.sidebar.slider('Choose a value for the n estimators parameter.', 1, 15, 1)
     min_samples_leaf = st.sidebar.slider('Choose a value for the min samples leaf parameter.', 1, 5, 1)
     max_depth = st.sidebar.slider('Choose a value for the max depth parameter.', 0, 3, 0)
     if max_depth == 0:
      max_depth = None
      pass
	
	
    if normalization == "Min Max Normalization":
        xtrain_normalized = MinMaxScaler().fit_transform(xtrain)
        xtest_normalized = MinMaxScaler().fit_transform(xtest)
    elif normalization == "Z-Score":
        xtrain_normalized = StandardScaler().fit_transform(xtrain)
        xtest_normalized = StandardScaler().fit_transform(xtest)
    else:
        xtrain_normalized = xtrain
        xtest_normalized = xtest
     pass

    #st.write(str(grid.score(xtrain_normalized, ytrain)))
    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(xtest_normalized)
    best_clf.fit(xtrain_normalized, y_train)
    predictions = best_clf.predict(xtest_normalized)
    st.write("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

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
	'classifier__min_samples_split': min_samples_split,
      	'classifier__max_depth': max_depth,
	'classifier__max_features': max_features
    	}
	    
     grid = GridSearchCV(pipe, parameters, cv=2).fit(xtrain_normalized, ytrain)
	
    #SVM
    elif "SVM" in selected_classifier:
     pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold()),
        ('classifier', SVC())
        ])

     pipe.fit(xtrain_normalized, ytrain)
    
     parameters = {
      	'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
	'selector__threshold': [0, 0.001, 0.01],
	'classifier__C': C,
      	'classifier__shrinking': shrinking,
        'classifier__probability': probability
    	}

     grid = GridSearchCV(pipe, parameters, cv=2).fit(xtrain_normalized, ytrain)

	
    #Adaboost
    elif "Adaboost" in selected_classifier:
     pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold()),
        ('classifier', AdaBoostClassifier())
        ])

     pipe.fit(xtrain_normalized, ytrain)
    
     parameters = {
      	'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
	'selector__threshold': [0, 0.001, 0.01],
	'classifier__n_estimators': n_estimators,
      	'classifier__learning_rate': learning_rate
    	}

     grid = GridSearchCV(pipe, parameters, cv=2).fit(xtrain_normalized, ytrain)

	
    #RandomForest
    else:
     pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold()),
        ('classifier', RandomForestClassifier())
        ])

     pipe.fit(xtrain_normalized, ytrain)
    
     parameters = {
      	'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
	'selector__threshold': [0, 0.001, 0.01],
	'classifier__n_estimators': n_estimators,
      	'classifier__min_samples_leaf': min_samples_leaf,
        'classifier__max_depth': max_depth
    	}

     grid = GridSearchCV(pipe, parameters, cv=2).fit(xtrain_normalized, ytrain)
	    
     pass
	
except ValueError as ve:
    print("")

