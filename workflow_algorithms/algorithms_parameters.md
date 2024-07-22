## Libraries
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import VotingClassifier

## **Previously defined variables**
 `best_model_1` = best model of algortihm 1  
 `best_model_2` = best model of algortihm 2  
 `model_1` =  scikit-learn model 1 library  
 `model_2` =  scikit-learn model 2 library
 
## **Support Vector Machine (SVM) - Parameters**
    svm_params = {
    'classifier__C': [1, 10, 100, 1000], 
    'classifier__gamma': [10, 1 ,0.1, 0.001],
    'classifier__kernel': ['linear', 'rbf', 'sigmoid']}

## **K-nearest Neighbors (KNN) - Parameters**
    knn_params = {
    'classifier__n_neighbors': list(range(1, 17, 2)),
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']}

## **Decision Tree (DT) - Parameters**
    dt_params = {
    'classifier__max_depth': [2, 3, 5, 7, 10], 
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__class_weight': ['balanced', None]}

## **Ensemble - Parameters**
    best_model_1= Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model_1(random_state=0, class_weight='balanced'))])

    best_model_2 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model_2(random_state=0, class_weight='balanced'))])

    voting_clf = VotingClassifier(
    estimators=[
        ('model_1', best_model_1),
        ('model_2', best_model_2)],
        voting='soft')
