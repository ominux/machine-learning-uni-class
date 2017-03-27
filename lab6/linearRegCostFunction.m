function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
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
h_theta = X * theta;

bias = 1 / (2 * m) * sum((h_theta - y) .^ 2);

theta1 = [0 ; theta(2:size(theta), :)];


Reg = lambda * sum(theta1 .^ 2) / (2 * m);

J = bias + Reg;

grad =  ( X' * (h_theta - y) + lambda*theta1 ) / m;


end