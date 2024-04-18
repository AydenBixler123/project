try:
    import streamlit as st
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    
	
    st.title("CIS 335 Project By: Ayden Bixler and Matthew Janatello")

    st.write("Please download the Titanic-Dataset.csv file from: www.kaggle.com/datasets/yasserh/titanic-dataset")
	
    uploadedfile = st.file_uploader("Please insert the Titanic-Dataset.csv file")

    df = pd.read_csv(uploadedfile)

    df = df.drop(['PassengerId', 'Embarked', 'Cabin', 'Ticket', 'Name', 'Sex'], axis=1)

    for ele in df:
        df.dropna(axis=0,inplace=True)

    st.write(df)
    
    train, test = train_test_split(df, test_size=0.2)
    
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    
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

    b=2
	
    ##Decision Tree
    if "Decision Tree" in selected_classifier:
     decision_tree_clf = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth, max_features=max_features)
     decision_tree_clf.fit(xtrain_normalized, ytrain)
     predictions = decision_tree_clf.predict(xtest_normalized)
     ascore = accuracy_score(ytest, predictions)
     pscore = cross_val_score(decision_tree_clf, xtest_normalized, ytest, scoring='precision')
     rscore = cross_val_score(decision_tree_clf, xtest_normalized, ytest, scoring='recall')
     f2score = ((1+b**2)*pscore*rscore)/(b**2 * pscore + rscore)
	    
    #SVM
    elif "SVM" in selected_classifier:
     SVC_clf = SVC(C=C, shrinking=shrinking, probability=probability)
     SVC_clf.fit(xtrain_normalized, ytrain)
     predictions = SVC_clf.predict(xtest_normalized)
     ascore = accuracy_score(ytest, predictions)
     pscore = cross_val_score(SVC_clf, xtest_normalized, ytest, scoring='precision')
     rscore = cross_val_score(SVC_clf, xtest_normalized, ytest, scoring='recall')
     f2score = ((1+b**2)*pscore*rscore)/(b**2 * pscore + rscore)
	     
    #Adaboost
    elif "Adaboost" in selected_classifier:
     Adaboost_clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
     Adaboost_clf.fit(xtrain_normalized, ytrain)
     predictions = Adaboost_clf.predict(xtest_normalized)
     ascore = accuracy_score(ytest, predictions)
     pscore = cross_val_score(Adaboost_clf, xtest_normalized, ytest, scoring='precision')
     rscore = cross_val_score(Adaboost_clf, xtest_normalized, ytest, scoring='recall')
     f2score = ((1+b**2)*pscore*rscore)/(b**2 * pscore + rscore)
	    
    #RandomForest
    else:
     RandomForest_clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
     RandomForest_clf.fit(xtrain_normalized, ytrain)
     predictions = RandomForest_clf.predict(xtest_normalized)
     ascore = accuracy_score(ytest, predictions)
     pscore = cross_val_score(RandomForest_clf, xtest_normalized, ytest, scoring='precision')
     rscore = cross_val_score(RandomForest_clf, xtest_normalized, ytest, scoring='recall')
     f2score = ((1+b**2)*pscore*rscore)/(b**2 * pscore + rscore)
     pass
	    
    st.write("Accuracy: " + str(ascore))
    st.write("Precision: " + str(pscore.mean()))
    st.write("Recall: " + str(rscore.mean()))
    st.write("F2Score: " + str(f2score))
	
except ValueError as ve:
    print("")

