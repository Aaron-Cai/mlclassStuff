function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.27;
sigma = 0.09;

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
% min = 1000;
% C0 = 0
% sig = 0
% for i = 0:7
%     for j = 0 : 7
%         model= svmTrain(X, y, C*(3^i), @(x1, x2) gaussianKernel(x1, x2, sigma*(3^j)));
%         error = mean(double(svmPredict(model,Xval)~=yval))
%         if error < min
%             min = error
%             C0 = C*(3^i);
%             sig = sigma*(3^j);
%         end
%     end
% end
% 
% C = C0
% sigma = sig







% =========================================================================

end
