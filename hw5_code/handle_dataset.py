import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

# All the constants are set in constants.py
from constants import *

scaler = StandardScaler()
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)


def basic_cleanup(input_csv):
    input_csv = input_csv.dropna(subset=[Y_index])
    string_types = set([x for x in input_csv.columns if input_csv.dtypes[x] == 'object'])
    if known_string_types != string_types:
        unknown_types = string_types - known_string_types
        print('Unhandled string attributes. ' + str(unknown_types) + '   Ignoring these attributes')
        input_csv = input_csv.drop(list(unknown_types), 1)
    # Text Values are mapped to corresponding numeric values. The numeric values are the indexes of the array listed
    # above.
    text_value_mapping = dict()
    for column in text_values.keys():
        local_value_mapping = dict(zip(text_values.get(column), range(len(text_values.get(column)))))
        text_value_mapping[column] = local_value_mapping
    input_csv = input_csv.replace(text_value_mapping)
    Y = np.array(input_csv[Y_index])
    input_csv = input_csv.drop(Y_index, 1)
    X = np.array(input_csv)
    return X, Y, list(input_csv.keys())


def load_data(input_csv):
    Xall,  Yall, words = basic_cleanup(input_csv)

    # Splitting the data before doing further cleanup.
    N, D = Xall.shape
    N0 = int(float(N) * 0.7)
    X = Xall[0:N0,:]
    Y = Yall[0:N0]
    Xte = Xall[N0:,:]
    Yte = Yall[N0:]
    X = imp.fit_transform(X)
    X = scaler.fit_transform(X)
    Xte = imp.transform(Xte)
    Xte = scaler.transform(Xte)
    return X, Y,  Xte, Yte, words


class peopleData:
    X, Y, Xte, Yte, words = load_data(pd.read_csv(input_file))