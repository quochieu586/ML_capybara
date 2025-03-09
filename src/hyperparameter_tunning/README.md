# Hyperparameter tunning using `optuna` library
## 1. Introduction to `optuna`
`optuna` is the Python library that use Bayesian Optimization for tunning hyperparameter. 

### How to use `optuna`
- The basic working unit of `optuna` is `optuna.study.Study` object, where it receive an **objective function**  as argument and execute this function trial by trial and return the argmax of **objective function**.
- In **objective function**, for each hyperparamter to be tunned, we need to created it via `optuna.trial.Trial` object, which is passed by default by `Study`.
- For each trial, the **objective function** suggest one value for each hyperparameter and return the evaluated value for those hyperparameters. Usually in each trial, we fit the model with those hyperparamters and evalute the model on validation test. 

## 2. Tunning pipeline
In this directory, we build an abstract class for tunning parameter for all models. The pipeline for using: 
1. Define hyperparamters to be tunned by defining `VariableSuggestion`, see **Section 3.1**.
2. Define training function and evaluation function. Training function will fit and return the model while evaluation function used to evaluate model performence in test set.
3. Create object `Tunning` and perform tunning, see **Section 3.2**. 

Example code with explaination:

```python 
from hyperparameter_tunning import *

# Construct suggestion instances
max_features_suggest = IntVariableSuggestion("max_features", 100, 1000)
max_gram_suggest = IntVariableSuggestion("max_grams", 2, 4)
nb_var_smooth_suggest = FloatVariableSuggestion("NB_var_smoothing", 1e-12, 1e-6)
tunning_params = [max_features_suggest, max_gram_suggest, nb_var_smooth_suggest]

# Define training function that receive train dataframe, hyperparamters and return the model
def training_func(train_df, max_features, max_grams, nb_var_smooth):
    # Extract TFIDF features
    features_vector =  feature_extraction.tfidf_vectorize(train_df['text'], max_features, max_grams)
    scaler = StandardScaler(with_mean=False)
    features_vector = scaler.fit_transform(features_vector)

    X_train = features_vector.toarray()
    y_train = train_df['label']

    # Evaluation hyperparamter
    model = GaussianNB(var_smoothing=nb_var_smooth)
    model.fit(X_train, y_train)
    return model

# Define evaluation model that receive testing dataframe, fitted model and return the result. The tunning objective function will try to maximize (or minimize, as defined in `direction`) this evaluation.
def evaluation_func(model, test_df, max_features, max_grams, nb_var_smooth):
    test_features_vector =  feature_extraction.tfidf_vectorize(test_df['text'], max_features, max_grams)
    X_test = test_features_vector.toarray()
    y_pred = model.predict(X_test)
    y_test = test_df['label']

    return f1_score(y_pred=y_pred, y_true=y_test, average='micro')

# This is our defined object for tunning hyperparameter
tunning_model = Tunning(tunning_params, training_func, evaluation_func)

# Using crossvalidation to evaluate on train data.
study = tunning_model.cross_validation_tunning(train_df, k_folds=15, n_trials=50, direction='maximize', timeout=120)
```


## 3. `hyperameter_tunning` directory in this Project

### 3.1 Variable suggestion
To tun the hyperparameters, we need to define the type of variables as well as the corresponding range. This is the role of `VariableSuggestion` class.

In this project, we support four kinds of variables:
- `IntVariableSuggestion(name, start, end)`: suggest int variable in range `[start, end]`.
- `FloatVariableSuggestion(name, start, end)`: suggest float variable in range `[start, end]`.
- `CategoricalVariableSuggestion(name, classes)`: suggest categorical variable from list `classes`.
- `ListVariableSuggestion(name, min_len, max_len, member: VariableSuggestion)`: suggest list of value, where each member is from the same suggestion class with `member`. Example:
```python
# Suggest number of layers and number of nodes in each layer for DNN model. 
# Number of layer is in range [2, 4]
# Number of nodes in each layer is in range [5, 10]  

layers_suggest = ListVariableSuggestion('layers', 2, 4, IntVariableSuggestion('', 5, 10))
tunning_params = [layers_suggest]

# Tunning model
def training_func(train_df, layers: List[int]):
    # Create DNN model which those layers
    # TODO HERE
    return model

# Evaluate model
def evaluation_func(model, test_df, layers):
    # TODO HERE
    return score

tunning_model = Tunning(tunning_params, training_func, evaluation_func)
study = tunning_model.cross_validation_tunning(train_df, k_folds=15, n_trials=50, direction='maximize', timeout=120)
```

### 3.2 Tunning
This is where tunning process happen. 
- We abstract the tunning process in which `Tunning` object receive **training function**, **evaluation function** and recieve list of hyperparamters and their corresponding suggestion value, which are defined in `VariableSuggestion` objects.
- We also construct two kinds of **objective function**
    + `tunning`: use `train_df` to fit model and `val_df` to evaluate model.
    + `cross_validation_tunning`: perform cross-validation in `df` to fit model in `k-1` folders and evaluate on the other folder.