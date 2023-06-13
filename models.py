from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse


def grid_search_cv_random_forest(X, Y):
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
