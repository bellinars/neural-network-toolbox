function [theta smOpt] = softmaxInit(inputData, smOpt)
% S. Mohsen Amiri (amiri1982@gmail.com) October 2012
% initialize a softmax model 
% 
% sm = softmaxInit(inputData, sm)
% sm:  a structure which contain the model parameters
% inputData: a MxN matrix which will be used for training. M is the data
%            dimensionality and N is the number of samples in the dataset 

if (~exist('smOpt','var'));
    smOpt = struct;
end
if (~isfield('smOpt', 'lambda'))
    smOpt.lambda = 1e-4; % Weight decay parameter
end

smOpt.inputDim = size(inputData,1);
theta = 0.005 * randn(smOpt.numClasses * smOpt.inputDim, 1);
end