import numpy as np
import pandas as pd
import sys
from sklearn import linear_model
from sklearn import preprocessing

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}


def polynomial_sframe(feature, degree):
    poly_dataset = pd.DataFrame()
    poly_dataset['power_1'] = feature
    if degree > 1:
        for power in range(2, degree + 1):
            column = 'power_' + str(power)
            poly_dataset[column] = feature ** power
    features = poly_dataset.columns.values.tolist()
    poly_dataset['constant'] = 1
    return poly_dataset, ['constant'] + features


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    derivative = 2 * feature.T.dot(errors)
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    if not feature_is_constant:
        derivative += 2 * l2_penalty * weight
    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations=100):
    weights = np.array(initial_weights).reshape((len(initial_weights), 1))  # make sure it's a numpy array
    iteration = 0
    old_RSS = sys.float_info.max
    rss_delta = sys.float_info.max
    # while iteration < max_iterations:
    while rss_delta > 1000:
        # while not reached maximum number of iterations:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        old_RSS = (errors.T.dot(errors))[0, 0]
        old_weights = np.copy(weights)
        rss_goes_down = False
        while not rss_goes_down:
            precalc_weights = np.copy(weights)
            for i in range(len(weights)):  # loop over each weight
                # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
                # compute the derivative for weight[i].
                # (Remember: when i=0, you are computing the derivative of the constant!)
                derivative = feature_derivative_ridge(errors, feature_matrix[:, i], old_weights[i, 0], l2_penalty,
                                                      i == 0)
                # subtract the step size times the derivative from the current weight
                precalc_weights[i, 0] -= step_size * derivative
            predictions = predict_output(feature_matrix, precalc_weights)
            precalc_errors = predictions - output
            RSS = (precalc_errors.T.dot(precalc_errors))[0, 0]
            if RSS > old_RSS:
                step_size /= 2
                print('Decreasing step size to %s. RSS: %s/%s' % (step_size, old_RSS, RSS))
            else:
                rss_goes_down = True
                weights = precalc_weights
                if iteration % 1000 == 0:
                    print('RSS diff: %s, RSS: %s' % (rss_delta, RSS))
                    step_size *= 1.2
                rss_delta = old_RSS - RSS
                old_RSS = RSS
        iteration += 1
    print('Ridge resgression completed in %s iterations with RSS diff: %s, RSS: %s' % (iteration, rss_delta, old_RSS))
    return weights


def predict_output(X, w):
    return X.dot(w)


sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(['sqft_living', 'price'])
sales['sqft_living_norm'] = (sales['sqft_living'] - sales['sqft_living'].mean()) / sales['sqft_living'].std()
# print(sales)
l2_small_penalty = 1e-5
poly_data, features = polynomial_sframe(sales['sqft_living_norm'], 1)
initial_weights = np.zeros((2, 1))
step_size = 1e-12
max_iterations = 10000000000
output = sales['price'].reshape((len(sales['price']), 1))
# weights = ridge_regression_gradient_descent(poly_data[features].values, output, initial_weights, step_size,
#                                            l2_small_penalty,
#                                            max_iterations=100)
# print(weights)
# [[  3.62161756e-30]
# [  8.90358643e-27]
# [  2.59708941e-23]
# [  8.49107815e-20]
# [  2.59714608e-16]
# [  3.55448944e-14]]


X = sales[['sqft_living', 'price']].values
sales_normalized = preprocessing.Normalizer(X, 'l2')
print(type(sales_normalized.transform(X)))

model = linear_model.Ridge(alpha=l2_small_penalty, normalize=False)
model.fit(sales_normalized[:, 0].reshape((len(sales_normalized[:, 0]), 1)), sales_normalized[:, 1])
print(model.coef_)
