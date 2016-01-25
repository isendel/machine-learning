import numpy as np
import pandas as pd
import math
from sklearn import linear_model

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)


def get_numpy_data(dataset, features, output_name):
    dataset['constant'] = 1
    return dataset[['constant'] + features].values, dataset[output_name].values.reshape(
            (len(dataset[output_name].values), 1))


def predict_output(feature_matrix, weights):
    return feature_matrix.dot(weights)


def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    return features / norms, norms


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    prediction = predict_output(feature_matrix, weights)
    ro_i = compute_roi(feature_matrix, i, output, prediction, weights)
    # print('Ro_%s: %s, l1_penalty / 2.: %s' % (i, ro_i, l1_penalty / 2.))
    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2.
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2.
    else:
        new_weight_i = 0
    return new_weight_i


def compute_roi(feature_matrix, i, output, prediction, weights):
    return sum(feature_matrix[:, i:i + 1] * (output - prediction + weights[i, 0] * feature_matrix[:, i:i + 1]))


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = initial_weights
    change_magnitude = np.zeros((len(weights), 1))
    converged = False
    cycle_count = 0
    while not converged:
        # new_weights = weights
        for i in range(len(weights)):
            weights_i = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            change_magnitude[i] = abs(weights[i, 0] - weights_i)
            # print('Change magnitude for %s: %s' % (i, change_magnitude[i]))
            # new_weights[i] = weights_i
            weights[i, 0] = weights_i
        # weights = new_weights
        magnitude_ = sum(change_magnitude)
        # print('Weights: %s' % weights)
        cycle_count += 1
        if cycle_count % 5 == 0:
            print('%s cycles passed. Magnitude: %s' % (cycle_count, magnitude_))
        if magnitude_ < tolerance:
            converged = True
    return weights


# initial_weights = np.array([1, 4, 1]).reshape((3,1))
# features_matrix, output = get_numpy_data(sales, ['sqft_living', 'bedrooms'], 'price')
# features_matrix_normalized, norm = normalize_features(features_matrix)
# prediction = predict_output(features_matrix_normalized, initial_weights)
# ro_0 = compute_roi(0, features_matrix_normalized, output, prediction, initial_weights)
# print(ro_0)
# ro_1 = compute_roi(1, features_matrix_normalized, output, prediction, initial_weights)
# print(ro_1)
# ro_2 = compute_roi(2, features_matrix_normalized, output, prediction, initial_weights)
# print(ro_2)




# features_matrix_all, output_all = get_numpy_data(sales, ['sqft_living', 'bedrooms'], 'price')
# normalized_feature_matrix, norm = normalize_features(features_matrix_all)
# print(norm)
# print(normalized_feature_matrix)
# l1_penalty = 1e7
# lasso_weights = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output_all, np.array([0, 0, 0]), l1_penalty,
#                                                   2)
# print(lasso_weights)
# errors = predict_output(normalized_feature_matrix, lasso_weights)
# rss = errors.T.dot(errors)
# print('RSS: %s' % rss)
#
# model = linear_model.Lasso(alpha=l1_penalty, normalize=False)
# model.fit(sales[['sqft_living', 'bedrooms']], sales['price'])
# print(model.coef_)
# print(model.intercept_)






train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')
all_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
                'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
train_features_matrix, train_output = get_numpy_data(train_data, all_features, 'price')
train_features_matrix_normalized, train_norm = normalize_features(train_features_matrix)
l1_penalty = 1e7
tolerance = 1
weights1e7 = lasso_cyclical_coordinate_descent(train_features_matrix_normalized, train_output,
                                               np.zeros((len(all_features) + 1, 1)),
                                               l1_penalty,
                                               tolerance)
print(weights1e7)
