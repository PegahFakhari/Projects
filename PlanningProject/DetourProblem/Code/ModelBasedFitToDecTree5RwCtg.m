% Model-Based RL fit/ In this file I assumed 5 categories
% for my rewards

clc
clear
close all
% ======== design the Environment ======== %
NumOfBlocks = 9;
TestPhaseNum = 8;
PrestestBlock = 7;
NumOfTrials = 20;
MaxNumPerEps = 15;
GridSize = [4, 7];
NumOfState = prod(GridSize);
NumOfAction = 4;
FixedLock = [1, 2, 4, 10, 18, 24, 25, 26, 28];
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
rW = 0;
rL = 0;
rG = 1;
r = 0;
RewStr = [rW, rG, r, rL];
Ifwhilenotmet = 200;
NumOfSubj  = 36; % 19 for Exp 1, 36 for Exp2, 32 for Exp 3
% ================================= %
% ========= MLE parameter =========== %
Discount = 0.95;
Temperature = 9;
alpha1_75 = 0;
alpha2_75 = 1;
alpha1_30 = 0;
alpha2_30 = 1;
alpha1_20 = 0;
alpha2_20 = 1;
alpha1_5 = 0;
alpha2_5 = 1;
alpha1_3 = 0;
alpha2_3 = 1;

Decay = 0;
BoundedZeroOne = 1;
if Decay == 1
    A = [Discount, Temperature, alpha1_75, alpha2_75,...
        alpha1_30, alpha2_30, alpha1_20, alpha2_20, alpha1_5, alpha2_5, alpha1_3, alpha2_3];
    fminresult = zeros(NumOfSubj, 13);
else
    A = [Discount, Temperature, alpha1_75, alpha1_30, alpha1_20, alpha1_5, alpha1_3];
    fminresult = zeros(NumOfSubj, 8);
end
IndFreeA = (1 : numel(A));

% option = optimset('Display','iter', 'MaxIter', 50);
option = optimset('Display','iter', 'TolX', 1e-004);
load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\InitValExp2'))

for iSub = 1 : NumOfSubj
    clc
    fprintf('sim=%d', iSub, NumOfSubj);
    ID = num2str(iSub);
    SubjectID = strcat('mdfs', ID);
    %     SubjectID = strcat('Exp1S', ID);
    %         SubjectID = strcat('STEN', ID, '-Whole'); %mdfs means I added the final position to the preypredatorposition
    load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\Exp 2\', SubjectID,'.mat'))
    if Decay == 1
        InitPar = [-1 - 5*rand, 5 + 10*rand,...
            rand, rand, rand, rand, rand,...
            rand, rand, rand, rand, rand];
    else
        %         InitPar = [InitVal(iSub, 1), InitVal(iSub, 2), ...
        %             InitVal(iSub, 3), InitVal(iSub, 4), InitVal(iSub, 5), 2];
        InitPar = [-1 - 5*rand, 5 + 10*rand, 6-rand, -rand, 5*rand, rand, rand];
    end
    [x,fval] = fminsearch( @(x) CalcLLRw5Ctg(x, NumOfTrials, NumOfBlocks, MaxNumPerEps, Ifwhilenotmet,...
        PreyPredatorPos, PreyPosAtEachAttempt, RightChoice, LeftChoice, UpChoice, DownChoice,...
        ImRewardData, ObsKindVecIn3Block, GridSize, NumOfAction, NumOfState, LearningRate_T,...
        Obs, PartialObs, pTrans4Obs, RewStr, CellsWStochRew, Decay, BoundedZeroOne, A, IndFreeA) , InitPar, option);
    % ImRewardData for subjects and ImRewardModel for simulation

    if Decay == 1
        fminresult(iSub, 1) = x(1);
        fminresult(iSub, 2) = x(2);
        fminresult(iSub, 3) = x(3);
        fminresult(iSub, 4) = x(4);
        fminresult(iSub, 5) = x(5);
        fminresult(iSub, 6) = x(6);
        fminresult(iSub, 7) = x(7);
        fminresult(iSub, 8) = x(8);
        fminresult(iSub, 9) = x(9);
        fminresult(iSub, 10) = x(10);
        fminresult(iSub, 11) = x(11);
        fminresult(iSub, 12) = x(12);
        fminresult(iSub, 13) = fval;
    else
        fminresult(iSub, 1) = x(1);
        fminresult(iSub, 2) = x(2);
        fminresult(iSub, 3) = x(3);
        fminresult(iSub, 4) = x(4);
        fminresult(iSub, 5) = x(5);
        fminresult(iSub, 6) = x(6);
        fminresult(iSub, 7) = x(7);
        fminresult(iSub, 8) = fval;
    end
    save(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\', ...
        strcat('fminfit5RCExp2'),'.mat'), 'fminresult')
end

hist(fminresult(:,2))
figure, hist(1./(1+exp(fminresult(:,1))))















