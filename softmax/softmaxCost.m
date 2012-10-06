function [cost, grad] = softmaxCost(theta, data, labels, smOpt)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, smOpt.numClasses, smOpt.inputDim);

numSamples = size(data, 2);

groundTruth = full(sparse(labels, 1:numSamples, 1));

M = theta*data;
M = bsxfun(@minus, M, max(M, [], 1)); % clip the data to zero

e = exp(M);
p = bsxfun(@rdivide,e,sum(e));
indicator = zeros(size(p));
indicator((0:size(labels,1)-1)'*(smOpt.numClasses)+labels) = 1;

cost = -(1/numSamples)*sum(sum(log(p).*indicator)) + 0.5*smOpt.lambda*sum(theta(:).^2);
thetagrad = -(1/numSamples)*(indicator-p)*data' +smOpt.lambda*(theta);

grad = thetagrad(:);
end

