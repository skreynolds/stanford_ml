function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

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

C_vec = zeros(1,64);
sigma_vec = zeros(1,64);
error_vec = zeros(1,64);
[C_mat sigma_mat] = ...
    meshgrid([0.01 0.03 0.1 0.3 1 3 10 30], [0.01 0.03 0.1 0.3 1 3 10 30]);

C_vec(:) = C_mat;
sigma_vec(:) = sigma_mat;

for i = 1:64
    
    % Train the parameters for the model using a specific C, sigma
    % combination
    model= ...
        svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(i)));
    
    % Obtain model predictions for the cross validation set
    pred = svmPredict(model,Xval);
    
    % Compute error
    error(i) = mean(double(pred ~= yval));
    
end

% Find the index which gives the minimum error
[Y I] = min(error);

C = C_vec(I)
sigma = sigma_vec(I)

% =========================================================================

end
