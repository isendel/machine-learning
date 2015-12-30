function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m,n] = size(X);

thetaPt = theta;
thetaPt(1)=0;

J = ((-1/m)*sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))))+(lambda/(2*m))*(thetaPt'*thetaPt)
grad = (1/m)*X'*(sigmoid(X*theta)-y)+(lambda/m)*thetaPt;

end
