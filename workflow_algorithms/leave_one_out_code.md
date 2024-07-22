## **Leave-One-Out strategy**

## **Libraries**  
    import numpy as np
    from sklearn.preprocessing import StandardScaler  
    from sklearn.model_selection import (train_test_split,
                                     GridSearchCV)
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import LeaveOneOut

## **Previously defined variables**
`model_params` = parameters used for cross-validation  
`model_clf` = algorithm  
`model` = algorithm in scikit-learn library  
`train` = $n$ total of elements, with $(n-1)$ for training  
`test` =  test from one ($1$) patient, cycling through all ($n$) patients

## **Implementation**

    model_params = {
    'parameter_1': [ ], 
    'parameter_2': [ ],
    'parameter_3': [ ],
    }

    model_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model(random_state=42, class_weight='balanced'))
    ])

    kf = LeaveOneOut()
    all_y = []
    all_probs_model = []
    all_preds_model = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        grid_search_model = GridSearchCV(model_clf, model_params, cv=10, scoring='roc_auc_ovr', n_jobs=-1)
        grid_search_model.fit(X_train, y_train)
        best_model = grid_search_model.best_estimator_
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
    
        all_y.extend(y_test.tolist())
        all_probs_model.extend(y_prob.tolist())
        all_preds_model.extend(y_pred.tolist())

    all_y = np.array(all_y)
    all_probs_model = np.array(all_probs_svm)
    all_preds_model = np.array(all_preds_svm)
