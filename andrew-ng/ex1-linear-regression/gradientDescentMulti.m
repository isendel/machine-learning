function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

[m,n] = size(X); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    delta = ones(n,1);
    for i = 1:n
      delta(i) = (1/m)*(X*theta - y)'*X(:,i);
    end
    
    theta = theta - alpha*delta;

    % Save the cost J in every iteration    
    J = computeCostMulti(X, y, theta);
    J_history(iter) = J;

end

end
