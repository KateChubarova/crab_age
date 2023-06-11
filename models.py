from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
from keras.layers import LeakyReLU, Dropout, Lambda
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from keras.models import Sequential


def grid_search_cv_random_forest(X,Y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=100)
    param_grid = {'n_estimators': [100, 200],
                  'min_samples_split': [8, 10], 'min_samples_leaf': [3, 4, 5],
                  'max_depth': [80, 90]}

    model = RandomForestRegressor()
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=-1,
                        scoring='neg_mean_absolute_error', refit=tree_classifier(), verbose=3)
    grid.fit(X, Y)
    print("grid search cv random forest")
    print(grid.best_params_)


def svc(X_train, y_train, X_test, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    print('svc')
    predictions = model.predict(X_test)
    print(mse(y_test, predictions))

def kneighbors_classifier(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print('kneighbors classifier')
    print(mse(y_test, predictions))

def kneighbors_regressor(X_train, y_train, X_test, y_test):
    model = KNeighborsRegressor(n_neighbors=15)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print('kneighbors regressor')
    print(mse(y_test, predictions))


def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=500, max_depth=100, min_samples_split=10, min_samples_leaf=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print('random_forest')
    print(mse(y_test, predictions))


def tree_classifier(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print('tree_classifier')
    print(mse(y_test, predictions))


def create_model(optimizer='adam', activation=Activation('relu'), dropout_rate=0.0, log_normalization=False):
    model = Sequential()

    if log_normalization:
        model.add(Lambda(lambda x: K.log(x + 1e-8)))  # Apply log transformation

    model.add(Dense(128, activation=activation, input_shape=(X.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_absolute_error')
    return model


def chat_gpt_keras_regressor_grid_search(X_train, y_train, X_test, y_test):
    keras_regressor = KerasRegressor(build_fn=create_model, verbose=0)
    param_grid = {
        'optimizer': ['adam', 'sgd'],
        'activation': [LeakyReLU(alpha=0.1), Activation('relu')],
        'dropout_rate': [0.0, 0.2, 0.4],
        'log_normalization': [False, True]
    }

    grid_search = GridSearchCV(estimator=keras_regressor, param_grid=param_grid, cv=5, verbose=3)
    grid_search.fit(X_train, y_train)

    print("Best parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)