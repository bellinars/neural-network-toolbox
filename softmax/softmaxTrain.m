function [softmaxModel] = softmaxTrain(data, labels, smOpt)

options.Method = 'lbfgs'; 

[theta0 smOpt]= softmaxInit(data, smOpt);
smCost = @(x)softmaxCost(x, data, labels, smOpt);

[theta_opt, ~] = minFunc( smCost, theta0, options);

softmaxModel.theta = theta_opt;
softmaxModel.smOpt = smOpt;

                          
end                          
