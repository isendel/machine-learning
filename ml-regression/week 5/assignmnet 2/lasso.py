import numpy as np
import pandas as pd
import math

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)


def get_numpy_data(dataset, features, output_name):
    dataset['constant'] = 1
    return dataset[['constant'] + features].values, dataset[output_name].values


def predict_output(feature_matrix, weights):
    return feature_matrix.dot(weights)


def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    return features / norms, norms


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    prediction = predict_output(feature_matrix, weights)
    ro_i = sum(feature_matrix[:, i] * (output - prediction + weights[i] * feature_matrix[:, i]))
    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2.
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2.
    else:
        new_weight_i = 0
    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = initial_weights
    change_magnitude = np.zeros((len(weights), 1))
    converged = False
    while not converged:
        new_weights = weights
        for i in range(len(weights)):
            weights_i = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            change_magnitude[i] = abs(weights[i] - weights_i)
            new_weights[i] = weights_i
            # weights[i] = weights_i
        weights = new_weights
        magnitude_ = sum(change_magnitude)
        print('Weights: %s' % weights)
        if magnitude_ < tolerance:
            converged = True
    return weights


features_matrix_all, output_all = get_numpy_data(sales, ['sqft_living', 'bedrooms'], 'price')
normalized_feature_matrix, norm = normalize_features(features_matrix_all)
print(norm)
print(normalized_feature_matrix)
lasso_weights = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output_all, np.array([0, 0, 0]), 1e7,
                                                  1)
print(lasso_weights)
