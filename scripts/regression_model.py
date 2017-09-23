import re
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

regex_fix_json = re.compile(r"((?!\{)\d+(?=:))")


def kfold_split(dataset, n_splits=5):
    kfold = KFold(n_splits, random_state=42)
    unique_sku = dataset['sku'].unique()

    for train_index, test_index in kfold.split(unique_sku):
        train_data = dataset[dataset['sku'].isin(unique_sku[train_index])]
        test_data = dataset[dataset['sku'].isin(unique_sku[test_index])]
        yield train_data, test_data


def features_extractor(dataset):
    return dataset['TEXT'], dataset['RATING']


def calculate_errors(y_true, y_pred):
    errors = dict()
    errors['mse'] = mean_squared_error(y_true, y_pred)
    return errors


def average_errors(errors_list):
    def average_by_key(errors, key):
        return sum(d[key] for d in errors) / float(len(errors))

    keys = errors_list[0].keys()
    result = {key: average_by_key(errors_list, key) for key in keys}
    return result


def train_model(train_data, extractor, model):
    X_train, y_train = extractor(train_data)
    model.fit(X_train, y_train)
    return model


def evaluate_model(test_data, extractor, model):
    X_test, y_test = extractor(test_data)
    prediction = model.predict(X_test)
    fold_errors = calculate_errors(y_test, prediction)
    return fold_errors


def evaluate_with_labels(test_data, extractor, model):
    X_test, y_test = extractor(test_data)
    prediction = model.predict(X_test)
    fold_errors = calculate_errors(y_test, prediction)
    return fold_errors, prediction


def train_and_evaluate(dataset, extractor, model):
    errors_list = list()

    for train_data, test_data in kfold_split(dataset):
        model = train_model(train_data, extractor, model)
        fold_errors = evaluate_model(test_data, extractor, model)
        errors_list.append(fold_errors)

    return average_errors(errors_list)


def prepare_json(data):
    data = data.replace('\'', '"')
    data = regex_fix_json.sub(r'"\1"', data)
    result_dict = dict()

    for item in json.loads(data):
        result_dict.update(item)

    return result_dict


def extract_properties(dataset):
    sku_properties = dict()

    for row in dataset[['sku', 'property']].drop_duplicates().itertuples():
        assert (row.sku not in sku_properties)
        sku_properties[row.sku] = prepare_json(row.property)

    return sku_properties


