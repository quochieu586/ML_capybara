import optuna
from VariableSuggest import *
from typing import List, Generator
import pandas as pd
from sklearn.model_selection import KFold

class Tunning:
    def __init__(self, tunning_params: List[VariableSuggestion], training_func: callable, evaluation_func: callable):
        self.tunning_params = tunning_params
        self.training = training_func
        self.evaluation = evaluation_func

    def tunning(self, train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int, direction: str = 'maximize', timeout: int = 120) -> optuna.study.Study:        
        def objective(trial: optuna.trial.Trial):
            params = []
            
            for hyperparam in self.tunning_params:
                value = hyperparam.get_suggest(trial)
                params += [value]
            
            model = self.training(train_df, *params)
            return self.evaluation(model, val_df, *params)
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        return study
    
    def cross_validation_tunning(self, df: pd.DataFrame, n_trials: int, k_folds: int = 5, direction: str = 'maximize', timeout: int = 120) -> optuna.study.Study:
        """
            Cross validation tunning.
            --- 
            df: Dataframe to be train the model
            k_folds: number of folds that used for cross validation
        """        
        def objective(trial: optuna.trial.Trial):
            params = []
            
            for hyperparam in self.tunning_params:
                value = hyperparam.get_suggest(trial)
                params += [value]
            
            # Cross validation
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            validation_res = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]

                model = self.training(train_df, *params)
                validation_res +=  [self.evaluation(model, val_df, *params)]

            return sum(validation_res) / len(validation_res)
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        return study