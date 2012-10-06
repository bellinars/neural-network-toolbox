function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta

theta = reshape(softmaxModel.theta, softmaxModel.smOpt.numClasses, ...
                softmaxModel.smOpt.inputDim);
    
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

p = theta*data;

[~, pred] = max(p);
pred = pred';



% ---------------------------------------------------------------------

end

