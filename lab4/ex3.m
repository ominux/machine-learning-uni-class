%% ML 2017 Lab work 4, Part 1: Multiclass classification (One-vs-all) with logistic regression
% 
%You will need to complete the following functions in this exericse:
%
%     oneVsAll.m
%     predictOneVsAll.m

%% Initialization
clear ; close all; clc

%% Setup the parameters 
num_labels = 10;% 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Load and Visualize Data =============
% Load Data
load('ex3data1.mat');

m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression code from the last exercise. 
%You task is to implement one-vs-all classification for the handwritten digit dataset 
%by completing oneVsAll.m function
%
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);


%% ================ Part 3: Predict for One-Vs-All ================
%Complete the code in predictOneVsAll.m to use the trained (one-vs-all)
%classifiers to make predictions for the training data

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(pred (:) == y(:)) * 100);

