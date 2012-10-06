% S. Mohsen Amiri (amiri1982@gmail.com) October 2012
% An example for testing an using softmax cost function and linear
% with softmax cost
% The code is based on Andrew Ng's online course
clc
clear
close all


%%======================================================================
%% STEP 1: Load data
%% DEBUG
DEBUG = false; % Set DEBUG to true when debugging.
if DEBUG
    M = 8;
    N = 100;
    smOpt.numClasses = 10;
    inputData = randn(M, N);
    labels = randi(smOpt.numClasses, N, 1);
    [theta smOpt]= softmaxInit(inputData, smOpt);
    smCost = @(x)softmaxCost(x, inputData, labels, smOpt);
    [cost, grad] = smCost(theta);
    %% STEP 3: Gradient checking
    numGrad = computeNumericalGradient( smCost, theta);
    
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);

    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    
    disp(diff); % The difference should be small.     
    % In our implementation, these values are usually less than 1e-7.
    return
end

load('dataset/MNIST_dataset.mat');
smOpt.numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

%%======================================================================
%% STEP 4: Learning parameters
%

smOpt.maxIter = 200;
softmaxModel = softmaxTrain(train.X, train.y, smOpt);
                          
%%======================================================================
%% STEP 5: Testing

[pred] = softmaxPredict(softmaxModel, test.X);

acc = mean(test.y(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% After 100 iterations, the results for our implementation were:
%
% Accuracy: 92.200%
%
% If your values are too low (accuracy less than 0.91), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
