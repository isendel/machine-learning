function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
values = [0.01 0.03 0.1 0.3 1 3 10 30];
C = 0;
sigma = 0;
error = 1000000;
for i = 1:size(values,2)
    for j = 1:size(values,2)
        Ctest = values(i);
        sigmaTest = values(j);
        model = svmTrain(X, y, Ctest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
        predictions = svmPredict(model, Xval);
        errorTest = mean(double(predictions ~= yval));
        if(errorTest < error)
            error = errorTest;
            C = Ctest;
            sigma = sigmaTest;
            fprintf('Found new C and sigma with smaller errror: C=%f, sigma=%f, error=%f', C,sigma, error);
        end
    end
end
fprintf('Result C and sigma with minimum errror: C=%f, sigma=%f, error=%f', C,sigma, error);


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% =========================================================================

end
