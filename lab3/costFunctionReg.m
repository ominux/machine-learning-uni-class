function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost J of a particular choice of theta.
% Note: grad should have the same dimensions as theta
%            
z = X*theta;

h= 1 ./ (  1 + exp(-z) );

J= (sum( -y.*log(h)-(1-y).*log(1-h))/m ) + ((lambda/(2*m)) *theta(2:end)'*theta(2:end) ); 

grad= ( (1/m) * ( h - y)' * X ) + (lambda/m).*theta' ;

grad(1)=  (1/m)*(h-y)'*X(:,1);
grad = grad(:);
% =============================================================
end
