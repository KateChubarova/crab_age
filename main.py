import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# check data for nulls, duplicates
# divide set to X and Y
# divide set to train and test
# use model
# validate
# get predictions for test
# upload to kaggle

data = pd.read_csv("train.csv")
X_submission = pd.read_csv('test.csv')

Y = data['Age']
X = data.drop(['Age', 'id'], axis='columns')


def gender_to_numeric(x):
    if x == 'I': return 3
    if x == 'M': return 2
    if x == 'F': return 1


X['Sex'] = X['Sex'].apply(gender_to_numeric)
X_submission['Sex'] = X_submission['Sex'].apply(gender_to_numeric)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)


def info():
    print(data.head())
    print(data.shape)
    print(data.dtypes)


def tree_classifier():
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print('tree_classifier')
    print(mse(y_test, predictions))


def write_to_file(predictions):
    output = pd.DataFrame({'id': X_submission.id, 'Age': predictions})
    output.to_csv('submission.csv', index=False)


def calculate_corr():
    corr_matrix = data.corr()
    print(corr_matrix["Age"].sort_values(ascending=False))


def random_forest():
    model = RandomForestRegressor(n_estimators=250, max_depth=20)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print('random_forest')
    print(mse(y_test, predictions))

    y_submission = model.predict(X_submission.drop('id', axis='columns'))
    write_to_file(y_submission)


def gradient_booster():
    model = GradientBoostingRegressor(n_estimators=300,
                                 learning_rate=0.05)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print('gradient booster')
    print(mse(y_test, predictions))

    y_submission = model.predict(X_submission.drop('id', axis='columns'))
    write_to_file(y_submission)


def kneighbors_regressor():
    model = KNeighborsRegressor(n_neighbors=15)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print('kneighbors regressor')
    print(mse(y_test, predictions))


def kneighbors_classifier():
    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print('kneighbors classifier')
    print(mse(y_test, predictions))


# random_forest()  # 4.370084275904191
# tree_classifier() #5.691819781478905
gradient_booster() #4.239210263601029 , 4.206711490036174(n_estimators=300,learning_rate=0.05, random_state=100)
# kneighbors_regressor() #4.447641236194659
# kneighbors_classifier() #5.915906207799648

# calculate_corr()