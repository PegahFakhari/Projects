% create a martix which contains all BIC for all subjects
clc
clear
close all
NumOfModels = 3;
Exp1No = 19;
Exp2No = 36;
Exp3No = 32;
Exp1BICModel = zeros(Exp1No, NumOfModels);
Exp1AICModel = zeros(Exp1No, NumOfModels);
Exp2BICModel = zeros(Exp2No, NumOfModels);
Exp2AICModel = zeros(Exp2No, NumOfModels);
Exp3BICModel = zeros(Exp3No, NumOfModels);
Exp3AICModel = zeros(Exp3No, NumOfModels);

% =========== Model 1 ============ %
% =========== Baseline ============ %
% load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\BaseLineModelResult\LLPredTest-Exp1-BaseLineModel')
% LLModel1Exp1 = SubjLLInTest';
% load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\BaseLineModelResult\LLPredTest-Exp2-BaseLineModel')
% LLModel1Exp2 = SubjLLInTest';
% load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\BaseLineModelResult\LLPredTest-Exp3-BaseLineModel')
% LLModel1Exp3 = SubjLLInTest';

% =========== Model 2 ============ %
% ===== Q Learning ===== %
% load('C:\Users\Pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelFreeQLearning\LLPredTest-Exp1-Qlearning')
% LLModel6Exp1 = SubjLLInTest';
% load('C:\Users\Pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelFreeQLearning\LLPredTest-Exp2-Qlearning')
% LLModel6Exp2 = SubjLLInTest';
% load('C:\Users\Pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelFreeQLearning\LLPredTest-Exp3-Qlearning')
% LLModel6Exp3 = SubjLLInTest';

% =========== Model 3 ============ %
% ===== ModelBased LinFil 3 Cat ===== %
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedLinFil3CatRew\LLPredTest-Exp1-3Alphas')
LLModel3Exp1 = SubjLLInTest';
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedLinFil3CatRew\LLPredTest-Exp2-3Alphas')
LLModel3Exp2 = SubjLLInTest';
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedLinFil3CatRew\LLPredTest-Exp3-3Alphas')
LLModel3Exp3 = SubjLLInTest';

% =========== Model 4 ============ %
% ===== Avoid 75 ===== %
% load('C:\Users\Pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedAvoid75\LLPredTest-Exp1-Avoid75')
% LLModel5Exp1 = SubjLLInTest';
% load('C:\Users\Pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedAvoid75\LLPredTest-Exp2-Avoid75')
% LLModel5Exp2 = SubjLLInTest';
% load('C:\Users\Pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedAvoid75\LLPredTest-Exp3-Avoid75')
% LLModel5Exp3 = SubjLLInTest';

% =========== Model 5 ============ %
% ============= LastR ============ %
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedLastR\LLPredTest-Exp1-LastR')
LLModel5Exp1 = SubjLLInTest';
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedLastR\LLPredTest-Exp2-LastR')
LLModel5Exp2 = SubjLLInTest';
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedLastR\LLPredTest-Exp3-LastR')
LLModel5Exp3 = SubjLLInTest';

% =========== Model 6 ============ %
% ========== ShortestPath ========= %
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedShortestPath\LLPredTest-Exp1-ShortestPath')
LLModel6Exp1 = SubjLLInTest';
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedShortestPath\LLPredTest-Exp2-ShortestPath')
LLModel6Exp2 = SubjLLInTest';
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedShortestPath\LLPredTest-Exp3-ShortestPath')
LLModel6Exp3 = SubjLLInTest';

% ====== Number of Observation ====== %
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\NumOfObsInBICExp1Rep');
NoObExp1 = NumOfObsPerSub;
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\NumOfObsInBICExp2Rep');
NoObExp2 = NumOfObsPerSub;
load('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\NumOfObsInBICExp3Rep');
NoObExp3 = NumOfObsPerSub;

% % % % % =========== Model 1 ============ %
% % % % % =========== Baseline ============ %
% % % % Model1NumOfParam = 0;
% % % % Exp1BICModel(:, 1) = 2 * LLModel1Exp1 + Model1NumOfParam .* log(NoObExp1)';
% % % % Exp1AICModel(:, 1) = 2 * LLModel1Exp1 + 2 * Model1NumOfParam;
% % % % 
% % % % Exp2BICModel(:, 1) = 2 * LLModel1Exp2 + Model1NumOfParam .* log(NoObExp2)';
% % % % Exp2AICModel(:, 1) = 2 * LLModel1Exp2 + 2 * Model1NumOfParam;
% % % % 
% % % % Exp3BICModel(:, 1) = 2 * LLModel1Exp3 + Model1NumOfParam .* log(NoObExp3)';
% % % % Exp3AICModel(:, 1) = 2 * LLModel1Exp3 + 2 * Model1NumOfParam;
% % % % % =========== Model 2 ============ %
% % % % % ===== Q Learning ===== %
% % % % Model2NumOfParam = 6;
% % % % Exp1BICModel(:, 2) = 2 * LLModel2Exp1 + Model2NumOfParam .* log(NoObExp1)';
% % % % Exp1AICModel(:, 2) = 2 * LLModel2Exp1 + 2 * Model2NumOfParam;
% % % % 
% % % % Exp2BICModel(:, 2) = 2 * LLModel2Exp2 + Model2NumOfParam .* log(NoObExp2)';
% % % % Exp2AICModel(:, 2) = 2 * LLModel2Exp2 + 2 * Model2NumOfParam;
% % % % 
% % % % Exp3BICModel(:, 2) = 2 * LLModel2Exp3 + Model2NumOfParam .* log(NoObExp3)';
% % % % Exp3AICModel(:, 2) = 2 * LLModel2Exp3 + 2 * Model2NumOfParam;
% =========== Model 3 ============ %
% ===== ModelBased LinFil 3 Cat ===== %
Model3NumOfParam = 5;
Exp1BICModel(:, 1) = 2 * LLModel3Exp1 + Model3NumOfParam .* log(NoObExp1)';
Exp1AICModel(:, 1) = 2 * LLModel3Exp1 + 2 * Model3NumOfParam;

Exp2BICModel(:, 1) = 2 * LLModel3Exp2 + Model3NumOfParam .* log(NoObExp2)';
Exp2AICModel(:, 1) = 2 * LLModel3Exp2 + 2 * Model3NumOfParam;

Exp3BICModel(:, 1) = 2 * LLModel3Exp3 + Model3NumOfParam .* log(NoObExp3)';
Exp3AICModel(:, 1) = 2 * LLModel3Exp3 + 2 * Model3NumOfParam;
% % % % =========== Model 4 ============ %
% % % % ===== Avoid 75 ===== %
% % % Model4NumOfParam = 2;
% % % Exp1BICModel(:, 4) = 2 * LLModel4Exp1 + Model4NumOfParam .* log(NoObExp1)';
% % % Exp1AICModel(:, 4) = 2 * LLModel4Exp1 + 2 * Model4NumOfParam;
% % % 
% % % Exp2BICModel(:, 4) = 2 * LLModel4Exp2 + Model4NumOfParam .* log(NoObExp2)';
% % % Exp2AICModel(:, 4) = 2 * LLModel4Exp2 + 2 * Model4NumOfParam;
% % % 
% % % Exp3BICModel(:, 4) = 2 * LLModel4Exp3 + Model4NumOfParam .* log(NoObExp3)';
% % % Exp3AICModel(:, 4) = 2 * LLModel4Exp3 + 2 * Model4NumOfParam;
% =========== Model 5 ============ %
% ============= LastR ============ %
Model5NumOfParam = 2;
Exp1BICModel(:, 2) = 2 * LLModel5Exp1 + Model5NumOfParam .* log(NoObExp1)';
Exp1AICModel(:, 2) = 2 * LLModel5Exp1 + 2 * Model5NumOfParam;

Exp2BICModel(:, 2) = 2 * LLModel5Exp2 + Model5NumOfParam .* log(NoObExp2)';
Exp2AICModel(:, 2) = 2 * LLModel5Exp2 + 2 * Model5NumOfParam;

Exp3BICModel(:, 2) = 2 * LLModel5Exp3 + Model5NumOfParam .* log(NoObExp3)';
Exp3AICModel(:, 2) = 2 * LLModel5Exp3 + 2 * Model5NumOfParam;
% =========== Model 6 ============ %
% ========== ShortestPath ========= %
Model6NumOfParam = 2;
Exp1BICModel(:, 3) = 2 * LLModel6Exp1 + Model6NumOfParam .* log(NoObExp1)';
Exp1AICModel(:, 3) = 2 * LLModel6Exp1 + 2 * Model6NumOfParam;

Exp2BICModel(:, 3) = 2 * LLModel6Exp2 + Model6NumOfParam .* log(NoObExp2)';
Exp2AICModel(:, 3) = 2 * LLModel6Exp2 + 2 * Model6NumOfParam;

Exp3BICModel(:, 3) = 2 * LLModel6Exp3 + Model6NumOfParam .* log(NoObExp3)';
Exp3AICModel(:, 3) = 2 * LLModel6Exp3 + 2 * Model6NumOfParam;
% % % % =========== Model 7 ============ %
% % % % ===== Windowed ModelBased Learning ===== %
% % % Model7NumOfParam = 6;
% % % Exp1BICModel(:, 7) = 2 * LLModel7Exp1 + Model7NumOfParam .* log(NoObExp1)';
% % % Exp1AICModel(:, 7) = 2 * LLModel7Exp1 + 2 * Model7NumOfParam;
% % % 
% % % Exp2BICModel(:, 7) = 2 * LLModel7Exp2 + Model7NumOfParam .* log(NoObExp2)';
% % % Exp2AICModel(:, 7) = 2 * LLModel7Exp2 + 2 * Model7NumOfParam;
% % % 
% % % Exp3BICModel(:, 7) = 2 * LLModel7Exp3 + Model7NumOfParam .* log(NoObExp3)';
% % % Exp3AICModel(:, 7) = 2 * LLModel7Exp3 + 2 * Model7NumOfParam;
% % % % =========== Model 8 ============ %
% % % Model8NumOfParam = 2;
% % % Exp1BICModel(:, 8) = 2 * LLModel8Exp1 + Model8NumOfParam .* log(NoObExp1)';
% % % Exp1AICModel(:, 8) = 2 * LLModel8Exp1 + 2 * Model8NumOfParam;
% % % 
% % % Exp2BICModel(:, 8) = 2 * LLModel8Exp2 + Model8NumOfParam .* log(NoObExp2)';
% % % Exp2AICModel(:, 8) = 2 * LLModel8Exp2 + 2 * Model8NumOfParam;
% % % 
% % % Exp3BICModel(:, 8) = 2 * LLModel8Exp3 + Model8NumOfParam .* log(NoObExp3)';
% % % Exp3AICModel(:, 8) = 2 * LLModel8Exp3 + 2 * Model8NumOfParam;


[alpha,exp_r,xp,pxpb1,bor] = bms((-.5) .* Exp1BICModel);
[alpha,exp_r,xp, pxpb2,bor] = bms((-.5) .* Exp2BICModel);
[alpha,exp_r,xp, pxpb3,bor] = bms((-.5) .* Exp3BICModel);
epbarBIC = [pxpb1; pxpb2; pxpb3];
bar(epbarBIC, 'group')
colormap(cool)

[alpha,exp_r,xp, pxpa1,bor] = bms((-.5) .* Exp1AICModel);
[alpha,exp_r,xp, pxpa2,bor] = bms((-.5) .* Exp2AICModel);
[alpha,exp_r,xp,pxpa3,bor] = bms((-.5) .* Exp3AICModel);
figure
epbarAIC = [pxpa1; pxpa2; pxpa3];
bar(epbarAIC, 'group')
colormap spring

figure
[ind, m] = min(Exp2AICModel');
hist(m)






