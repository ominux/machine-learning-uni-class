%% ML lab work -Unregularized  Logistic Regression
% 
%You will need to complete the following functions in this exericse:
%
%     plotData.m
%     costFunction.m
%     predict.m

%% Initialization
clear ; close all; clc

%% Step 1: Load and Plot Data from file ex2data1.txt
%  The first two columns contains the exam scores and the third column
%  contains the label.

fprintf('Loading data ...\n');

%% Step 1: Load Data from file ex2data1.txt
data = load('ex2data1.txt');
X = data(:,1:2);
y = data(:,3);
m = length(y); % number of training examples

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);


hold off;

%% ============ Part 2: Compute Cost and Gradient ============
%  Now you will implement the cost and gradient for logistic regression. 
%You neeed to complete the code in costFunction.m

%  Setup the data matrix appropriately
[m, n] = size(X);

% Add extra column of 1 to X
X =  [ones(m,1) X];

% Initialize fitting parameters =0
initial_theta = zeros(size(X,2) ,1);
% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);


%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
initial_theta = zeros(size(X,2) ,1);
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;

% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
z = [1 45 85]*theta;
prob = 1 ./ (  1 + exp(-z) )  ;
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);

%  Your task is to complete the code in predict.m
p= predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p== y)) * 100);

