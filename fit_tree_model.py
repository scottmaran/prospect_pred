import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import pickle

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
from xgboost import plot_importance

''' 
Fits XGBoost model and returns classifier and dataset (appended with predictions)
'''
def fit_model(save=True):
    dataset = pd.read_csv("model_data/input_dataset.csv", index_col=0)

    # drop targets and NFL production
    X = dataset.drop(["Score", "Success"] + ['num_seasons', 'GamesPlayed', 'GamesStarted', 'Plays', 'PositivePlays',
        'NegativePlays', 'GP%', 'GS%', 'PosPlay%', 'NegPlay%', 'NeutPlay%'],axis=1)
    X["ProPosition"] = X["ProPosition"].astype("category")
    X["IndyInvite"] = X["IndyInvite"].astype("category")

    def objective(trial):
        param = {"max_depth":    trial.suggest_categorical('max_depth', [2, 3, 4, 5, 6, 7, 8, 9]),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                "n_estimators": trial.suggest_int('n_estimators', 100, 1000,step=100),
                "subsample" : trial.suggest_float('subsample', 0.1, 1, step=0.1),
                "min_child_weight" : trial.suggest_int('min_child_weight', 1, 10, step=1), 
                "colsample_bytree" : trial.suggest_float('subsample', 0.1, 1, step=0.1),
                }
        
        clf = xgb.XGBRegressor(tree_method="hist", enable_categorical=True, 
                                        objective='reg:absoluteerror', **param)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, dataset['Score'], cv=kfold, scoring='neg_mean_absolute_error')
        score = np.mean(scores)
        return score
    # use optuna to find best hyperparameters for model
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    # fit model
    clf = xgb.XGBRegressor(tree_method="hist", enable_categorical=True, 
                                      objective='reg:squarederror', **study.best_params)
    clf.fit(X, dataset['Score'])
    # create predictions from LOOV
    model_pred = cross_val_predict(clf, X, dataset['Score'], cv=X.shape[0])

    dataset_with_scores = dataset.copy(deep=True)
    dataset_with_scores['Pred_Score'] = model_pred
    dataset_with_scores['Pred_Error'] = dataset_with_scores.Score - dataset_with_scores.Pred_Score
    dataset_with_scores = dataset_with_scores[['Pred_Score', 'Pred_Error'] + list(dataset.columns)]

    if save:
        pickle.dump(clf, open("model_data/xgb_model.pkl", "wb"))
        dataset_with_scores.to_csv("data/dataset_with_preds.csv")

    return clf, dataset_with_scores