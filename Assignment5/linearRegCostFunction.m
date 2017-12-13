function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predictions = X * theta; % predictions of hypothesis on all m examples
errors = (predictions - y);

J = sum(errors .^ 2) / (2*m);

% Compute Regularization

thetaMinus = theta(2:end,:);
reg = sum(thetaMinus .* thetaMinus);

J = J + (lambda * reg) / (2*m);

% Compute gradient with regularization

for j = 1 : length(grad)
    for i = 1 : m
        grad(j) = grad(j) + (1/m) * errors(i) * X(i, j);
    end
    
    if (j > 1)
        grad(j) = grad(j) + lambda / m * theta(j);
    end
end

% =========================================================================

grad = grad(:);

end
