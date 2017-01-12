% fit Model-Based RL with no learning for R fcn
clc
clear
close all
% ======== design the Environment ======== %
NumOfBlocks = 10;
TestPhaseNum = 8;
PrestestBlock = 7;
NumOfTrials = 20;
MaxNumPerEps = 15;
GridSize = [4, 7];
NumOfState = prod(GridSize);
NumOfAction = 4;
FixedLock = [1, 2, 4, 10, 18, 24, 25, 26, 28]; % !!#########!!
% FixedLock = [1, 2, 4, 10, 14, 18, 24, 25, 26, 28]; %For Exp 1
NoUpAvail = [12, 16];
NoDownAvail = [11, 15];
CheckAvailActInBl3 = 8;
Back2ReplanningInBl3 = 7;
LearningRate_T = 0.65;
LearningRate_R = 0.16;
Obs = 0;
PartialObs = [0; 0; 0; 0];
pTrans4Obs = [1, 1];
CellsWStochRew = [0; 0];
rW = -1;
rL = -1;
rG = 100;
r = -1;
RewStr = [rW, rG, r, rL];
Ifwhilenotmet = 200;
NumOfSubj  = 32; % 19 for Exp 1, 36 for Exp2, 32 for Exp 3
% ================================= %
% ========= MLE parameter =========== %
Discount = 0.95;
Temperature = 9;

A = [Discount, Temperature];
IndFreeA = (1 : numel(A));
% option = optimset('Display','iter', 'MaxIter', 50);
option = optimset('Display','iter', 'TolX', 1e-004);

fminresult = zeros(NumOfSubj, 3);

for iSub = 1 : NumOfSubj
    clc
    fprintf('sim=%d', iSub, NumOfSubj);
    
    ID = num2str(iSub);
%     SubjectID = strcat('mdfs', ID);
    %     SubjectID = strcat('Exp1S', ID);
    SubjectID = strcat('STEN', ID, '-Whole'); %mdfs means I added the final position to the preypredatorposition
    load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\Exp 3\', SubjectID,'.mat'))
    
%     InitPar = [-1 - 5*rand, 5 + 10*rand];
InitPar = [-1.233, 10.67];
    %       InitPar = [.8 + .2 *rand, 5 + 5*rand];
    [x,fval] = fminsearch( @(x) CalcLLWithLastR(x, NumOfTrials, NumOfBlocks, MaxNumPerEps, Ifwhilenotmet,...
        PreyPredatorPos, PreyPosAtEachAttempt, RightChoice, LeftChoice, UpChoice, DownChoice,...
        ImRewardData, ObsKindVecIn3Block, GridSize, NumOfAction, NumOfState, LearningRate_T,...
        Obs, PartialObs, pTrans4Obs, RewStr, CellsWStochRew, A, IndFreeA) , InitPar, option);
    % ImRewardData for subjects and ImRewardModel for simulation
    fminresult(iSub, 1) = x(1);
    fminresult(iSub, 2) = x(2);
    fminresult(iSub, 3) = fval;
    save(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\', ...
        strcat('fminfit'),'.mat'), 'fminresult')
end
hist(fminresult(:,2))
figure, hist(1./(1+exp(fminresult(:,1))))















