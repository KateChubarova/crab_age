import matplotlib.pyplot as plt
import pandas as pd


def gender_to_numeric(x):
    if x == 'I': return 3
    if x == 'M': return 2
    if x == 'F': return 1


def info(data):
    print(data.head())
    print(data.shape)
    print(data.dtypes)


def write_to_file(predictions, X_submission):
    output = pd.DataFrame({'id': X_submission.id, 'Age': predictions.reshape(len(predictions))})
    output.to_csv('submission.csv', index=False)


# created by Nikitosik (not my brilliant idea)
def results_nick_auth(predictions):
    X_submission = pd.read_csv('test.csv')
    output = pd.DataFrame({'id': X_submission.id, 'Age': predictions['Age']})
    output.to_csv('submission.csv', index=False)


def calculate_corr(data):
    corr_matrix = data.corr()
    print(corr_matrix["Age"].sort_values(ascending=False))


def feature_importances(model, X):
    importances = model.feature_importances_
    feature_names = X.columns
    plt.bar(x=feature_names, height=importances)
    plt.xlabel = "Feature importances using MDI"
    plt.ylabel = "Mean decrease in impurity"
    plt.show()


def remove_out_liners(X, y):
    print(X.columns.values.tolist())
    features = X.columns.values.tolist()
    for feature in features:
        Q1 = X[feature].quantile(0.15)
        Q3 = X[feature].quantile(0.85)
        IQR = Q3 - Q1
        outlier_threshold = 1.5
        outliers = X[(X[feature] < Q1 - outlier_threshold * IQR) | (X[feature] > Q3 + outlier_threshold * IQR)]
        X = X.drop(outliers.index)
        y = y.drop(outliers.index)
        print("Indices of removed outliers:", feature, outliers.index)
        return X, y
