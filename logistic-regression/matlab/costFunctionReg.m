function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m,n] = size(X);

% You need to return the following variables correctly 
grad = zeros(size(theta));

thetaPt = theta;
thetaPt(1) = 0;

J = ((-1/m)*sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))))+(lambda/(2*m))*(thetaPt'*thetaPt)
grad(1) = (1/m)*(sigmoid(X*theta)-y)'*X(:,1);
for i = 2:n
    grad(i) = (1/m)*(sigmoid(X*theta)-y)'*X(:,i)+(lambda/m)*theta(i);
end


end
