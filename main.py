import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
import numpy as np

# check data for nulls, duplicates
# divide set to X and Y
# divide set to train and test
# use model
# validate
# get predictions for test
# upload to kaggle

from helpers import gender_to_numeric, write_to_file, remove_out_liners

data = pd.read_csv("train.csv")
X_submission = pd.read_csv('test.csv')

Y = data['Age']
X = data.drop(['Age', 'id'], axis='columns')

X['Sex'] = X['Sex'].apply(gender_to_numeric)
X_submission['Sex'] = X_submission['Sex'].apply(gender_to_numeric)

# X = pd.get_dummies(X, columns=['Sex'])
# X_submission = pd.get_dummies(X_submission, columns=['Sex'])

X, y = remove_out_liners(X, Y)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)


def gradient_booster():
    model = GradientBoostingRegressor(n_estimators=500,
                                      learning_rate=0.01,
                                      subsample=0.7, max_depth=7)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print('gradient booster')
    print(mae(y_test, predictions))

    y_submission = model.predict(X_submission.drop('id', axis='columns'))
    write_to_file(y_submission, X_submission)


def grid_search_cv_gradient_boost():
    param_grid = {'n_estimators': [10, 50, 100, 300],
                  'learning_rate': [0.001, 0.01, 0.1, 1.0],
                  'subsample': [0.5, 0.7, 1.0],
                  'max_depth': [3, 7, 9, 15]}
    model = GradientBoostingRegressor()
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error')
    grid.fit(X, Y)
    print("grid search cv")
    print(grid.best_params_)


def get_model(X_train_normalized):
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train_normalized.shape[1],)))
    model.add(Dense(64, activation=LeakyReLU(alpha=0.1)))
    model.add(Dense(64, activation=LeakyReLU(alpha=0.1)))
    model.add(Dense(1, activation=LeakyReLU(alpha=0.1)))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def chat_gpt_keras_test():
    # X_train_normalized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    model = get_model(X_train)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.33)

    # X_test_normalized = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    y_prediction = model.predict(X_test)
    # output = pd.Series({'id': X_test_normalized.id, 'Age': y_prediction.reshape(len(y_prediction))})
    y_prediction = np.round(y_prediction)
    print(mae(y_prediction, y_test))

    _X_submission = X_submission.drop('id', axis='columns')
    # X_submission_normalized = (_X_submission - np.mean(_X_submission, axis=0)) / np.std(_X_submission, axis=0)
    y_submission = model.predict(_X_submission)
    #
    write_to_file(np.round(y_submission), X_submission)


chat_gpt_keras_test()  # 1.3891355250165365, 1.3824723439584041 - remove outliners

# с нормализацией
# без нормализации
# без нормолизации и без удаления выбросов
