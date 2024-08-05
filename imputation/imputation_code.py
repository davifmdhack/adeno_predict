## **Imputation process**

## **Libraries**
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    import random
    from fancyimpute import IterativeImputer
    from sklearn.impute import KNNImputer
    from sklearn.linear_model import LogisticRegression

## **Previously defined variables**
 `df` = dataset prior to imputation

## **Split train/test**
    best_seeds = []
    small_missing = float('inf')

    for seed in range(1000):
        train, test = train_test_split(df, 
                                   test_size=0.3, 
                                   random_state=seed)
        missing_teste = test.isna().sum().sum()
    
    if missing_teste < small_missing:
        best_seeds = [seed]
        small_missing = missing_teste
    elif missing_teste == small_missing:
        best_seeds.append(seed)
        
    random.seed(34) 
    melhor_seed_escolhida = best_seeds[0]
    chosen_seed = random.choice(best_seeds) # 5006 chosen seed

## **Missing variables in train set**
    train, test = train_test_split(df_encoded, 
                               test_size=0.3, 
                               random_state=5006)
    missing_teste = test.isna().sum()
    missing_train = train.isna().sum()

    print(f'Missing values ​​in training:\n{missing_train}')
    print(f'Missing values in seed test {chosen_seed}:\n{missing_teste}')

## **A. Imputation over `ADC` using MICE (logistic regression) - Stochastic process**
    imputer_1 = IterativeImputer(max_iter = 5, random_state=0, estimator=LogisticRegression())
    train[['adc']] = imputer_1.fit_transform(train[['adc']])
    
## **B. Imputation over `consistency` using KNN - Deterministic process**
    imputer_2 = KNNImputer(n_neighbors = 5)
    train[['consistency']] = imputer_2.fit_transform(train[['consistency']])
    train['consistency'] = np.round(train['consistency']).astype(int)

## **Confirmation of imputation and export of the final dataframe**
    print(train['consistency'].value_counts())
    print(test['consistency'].value_counts())
    df_set_final = pd.concat([train, test], axis = 0)
    df_set_final.to_excel('df_set_final.xlsx', sheet_name = 'Data Frame Imputation', index = False)
  
