import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = pd.DataFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree + 1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature ** power

    # poly_sframe['constant'] = 1
    return poly_sframe


def fit_poly(data_set, degree):
    poly_data_ = polynomial_sframe(data_set['sqft_living'], degree)
    features_ = poly_data_.columns.values[:degree].tolist()
    # features_ = ['constant'] + poly_data_.columns.values[:degree].tolist()
    model_ = linear_model.LinearRegression()
    # model_ = linear_model.LinearRegression(fit_intercept=False)
    model_.fit(poly_data_[features_], data_set['price'])
    return model_, poly_data_, features_


dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}
training_data = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation_data = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)

validation_RSS = []
models = []
features_list = []
for degree_value in range(1, 15 + 1):
    print('Learning degree %s ' % degree_value)
    model, poly_data, features = fit_poly(training_data, degree_value)
    print(features)
    validation_poly = polynomial_sframe(validation_data['sqft_living'], degree_value)
    validation_error = validation_data['price'] - model.predict(validation_poly[features])
    validation_RSS.append(validation_error.T.dot(validation_error))
    models.append(model)
    features_list.append(features)
print(validation_RSS)
best_degree = validation_RSS.index(min(validation_RSS))
print('Min validation error degree: %s' % (best_degree + 1))

