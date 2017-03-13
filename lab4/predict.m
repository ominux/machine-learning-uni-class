function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% NUmber of examples
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%

a1 = [ones(m,1) X];
z2= a1 * Theta1' ;
a2 = [ones(size(z2),1) , (1./(1+exp(-z2))) ];
z3=  a2 * Theta2';
a3 = (1./(1+exp(-z3)));

[~,ind]=max(a3');
p=ind;
% =========================================================================
end
