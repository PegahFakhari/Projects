% Simulate Subjects' Environment and choices.
clc
clear
close all

%===determine the directory===
a= 'FitWithMLE\RecoverParamDisSimulationModelBased.m';
Directory = which(a);
Directory = Directory(1:(end-numel(a)));
%===determine the directory===
SubjectEnv = 1;
if SubjectEnv ==1
    % recall from subject: LossFreqInPathA, LossFreqInPathB
    % LossFreqInPathC, ObsInEnvMat, ObsKindVecIn3Block
    load('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\EnvS27')
    Subj_LossFreqInPathA = LossFreqInPathA;
    Subj_LossFreqInPathB = LossFreqInPathB;
    Subj_LossFreqInPathC = LossFreqInPathC;
    Subj_ObsInEnvMat = ObsInEnvMat;
    Subj_ObsKindVecIn3Block = ObsKindVecIn3Block;
end
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
NoUpAvail = [12, 16];
NoDownAvail = [11, 15];
CheckAvailActInBl3 = 8;
Back2ReplanningInBl3 = 7;

% ============= Initial points ============ %
initpoint
% ============ Payofffs ============= %
ObsProbDis = [0.0001, .00000001, 1];
RewardProbDis = 0.8;
PathA1GreatLossAt = 11;
PathA2GreatLossAt = 19;
PathAGreatLossAt = [PathA1GreatLossAt, PathA2GreatLossAt];
PathBGreatLossAt = 16;
PathC1GreatLossAt = 9;
PathC2GreatLossAt = 17;
PathCGreatLossAt = [PathC1GreatLossAt, PathC2GreatLossAt];
ValueOfLossInPathA1 = -75;
ValueOfLossInPathA2 = -5;
ValueOfLossInPathB = -3;
ValueOfLossInPathC1 = -20;
ValueOfLossInPathC2 = -30;
% ==== Creating a Probabilistic Reward Function ====%
% ================ Path A ================== %
NumOfLossInA = RewardProbDis * NumOfBlocks * NumOfTrials;
VectA1 = [ValueOfLossInPathA1 * ones(1, NumOfLossInA),...
    (-1) * ones(1, NumOfBlocks * NumOfTrials - NumOfLossInA)];
VectA2 = [ValueOfLossInPathA2 * ones(1, NumOfLossInA),...
    (-1) * ones(1, NumOfBlocks * NumOfTrials - NumOfLossInA)];
% ================ Path B ================ %
NumOfLossInB = RewardProbDis * NumOfBlocks * NumOfTrials;
VectB = [ValueOfLossInPathB * ones(1, NumOfLossInB),...
    (-1) * ones(1, NumOfBlocks * NumOfTrials - NumOfLossInB)];
% ================ Path C ================ %
NumOfLossInC = RewardProbDis * NumOfBlocks * NumOfTrials;
VectC1 = [ValueOfLossInPathC1 * ones(1, NumOfLossInC),...
    (-1) * ones(1, NumOfBlocks * NumOfTrials - NumOfLossInC)];
VectC2 = [ValueOfLossInPathC2 * ones(1, NumOfLossInC),...
    (-1) * ones(1, NumOfBlocks * NumOfTrials - NumOfLossInC)];

% Initial positions in Test Phase based on the ObsKind %
InitPosVecIn3B = zeros(40, 2);
InitPosVecIn3B(:, 2) = 27;
InitPosVecIn3B(:, 1) = 3;
Check4Learning = [3, 27; 3, 27; 3, 27; 3, 27; 3, 27];
PretestInitPoints
GeneratePaths
% ==================================== %
% ==================================== %

% ========= Simulation parameters ========= %
Discount = 1/(1+exp(-1.3471));
Temperature = 10.5424;
LearningRate_T = 0.65;
LearningRate_R = 0.16;
NumOfSimulation  = 100;

% =================================== %
for iSim = 1 : NumOfSimulation
    clc
    fprintf('sim=%d', iSim);

    % =================================== %
    % ============ Initial points ============= %
    NumOfInitPos = size(InitPosVec, 1);
    IndexOfInitPos = randperm(NumOfInitPos);
    InitPosVecInBlocks = zeros(size(InitPosVec, 1), size(InitPosVec, 2));
    for s = 1 : NumOfInitPos
        Sindex = IndexOfInitPos(s);
        InitPosVecInBlocks(s,:) = InitPosVec(Sindex, :);
    end
    PreyPredatorPos = [InitPosVecInBlocks; Check4Learning; PretestPos; InitPosVecIn3B];
    PreyPosAtEachAttempt = zeros(MaxNumPerEps + 1, NumOfBlocks * NumOfTrials);
    RightChoice = zeros(NumOfBlocks, NumOfTrials, MaxNumPerEps);
    LeftChoice = zeros(NumOfBlocks, NumOfTrials, MaxNumPerEps);
    UpChoice = zeros(NumOfBlocks, NumOfTrials, MaxNumPerEps);
    DownChoice = zeros(NumOfBlocks, NumOfTrials, MaxNumPerEps);

    % ============== Payofffs ============== %
    % ============== Path A =============== %
    LossFreqInPathA1 = zeros(1, numel(VectA1));
    RandVectInPathA1 = randperm(numel(VectA1));
    for f = 1 : numel(RandVectInPathA1)
        Findex = RandVectInPathA1(f);
        LossFreqInPathA1(f) = VectA1(Findex);
    end
    LossFreqInPathA2 = zeros(1, numel(VectA2));
    RandVectInPathA2 = randperm(numel(VectA2));
    for f = 1 : numel(RandVectInPathA2)
        Findex = RandVectInPathA2(f);
        LossFreqInPathA2(f) = VectA2(Findex);
    end
    LossFreqInPathA = [LossFreqInPathA1; LossFreqInPathA2];

    % ============== Path B ============== %
    LossFreqInPathB = zeros(1, numel(VectB));
    RandVectInPathB = randperm(numel(VectB));
    for f = 1 : numel(RandVectInPathB)
        Findex = RandVectInPathB(f);
        LossFreqInPathB(f) = VectB(Findex);
    end

    % ============== Path C ============== %
    LossFreqInPathC1 = zeros(1, numel(VectC1));
    RandVectInPathC1 = randperm(numel(VectC1));
    for f = 1 : numel(RandVectInPathC1)
        Findex = RandVectInPathC1(f);
        LossFreqInPathC1(f) = VectC1(Findex);
    end
    LossFreqInPathC2 = zeros(1, numel(VectC2));
    RandVectInPathC2 = randperm(numel(VectC2));
    for f = 1 : numel(RandVectInPathC2)
        Findex = RandVectInPathC2(f);
        LossFreqInPathC2(f) = VectC2(Findex);
    end
    LossFreqInPathC = [LossFreqInPathC1; LossFreqInPathC2];

    % ============= Obstacles ============= %
    ObsKind = [(repmat (randperm (3), [1, 13])), 3];
    ObsKindInd = randperm(numel(ObsKind));
    ObsKindVecIn3Block = zeros(1, numel(ObsKind));
    for f = 1 : numel(ObsKind)
        Findex = ObsKindInd(f);
        ObsKindVecIn3Block(f) = ObsKind(Findex);
    end
    ObsInEnvMat = ObsGenFunc (FixedLock, GridSize, ObsProbDis,...
        NumOfBlocks, NumOfTrials, ObsKindVecIn3Block, PreyPredatorPos, PathSeq);

    % ===================================== %
    % ===================================== %
    % If Subject's environment is selected for similation %
    if SubjectEnv ==1
        LossFreqInPathA = Subj_LossFreqInPathA;
        LossFreqInPathB = Subj_LossFreqInPathB;
        LossFreqInPathC = Subj_LossFreqInPathC;
        ObsInEnvMat = Subj_ObsInEnvMat;
        ObsKindVecIn3Block = Subj_ObsKindVecIn3Block;
    end

    NumStepVec = zeros(1, NumOfBlocks * NumOfTrials);
    ImRewardModel = zeros(MaxNumPerEps, NumOfBlocks * NumOfTrials);
    % ================================= %
    % Generate the RewardFunc and ProbOfTransition
    Obs = 0;
    PartialObs = [0; 0; 0; 0];
    pTrans4Obs = [1, 1];
    CellsWStochRew = [0; 0];
    rW = -1;
    rL = -1;
    rG = 100;
    r = -1;
    RewStr = [rW, rG, r, rL];
    NumOfObser = zeros(NumOfState, NumOfState, NumOfAction);
    ObservedCell = zeros(NumOfState, NumOfAction);

    FirstGoalPoint = PreyPredatorPos(1, 2);
    [ProbOfTransition, RewardFunc] = ...
        ProbNRewFCNRandomWalkFormat(NumOfAction, GridSize, FirstGoalPoint, Obs, PartialObs, pTrans4Obs, RewStr, CellsWStochRew);
    GoalInR1 = (1:  4: 25);
    GoalInR4 = (4:  4: 28);
    if find(FirstGoalPoint == GoalInR1)
        QBeforeGoal = [0, -1, -1, -1; -1, -1, -1, -1];
        PBeforeGoal = [0, 1, 1, 1; 1, 1, 1, 1];
    else
        if find(FirstGoalPoint == GoalInR4)
            QBeforeGoal = [-1, -1, 0, -1; -1, -1, -1, -1];
            PBeforeGoal = [1, 1, 0, 1; 1, 1, 1, 1];
        else
            QBeforeGoal = [-1, -1, -1, -1; -1, -1, -1, -1];
            PBeforeGoal = [1, 1, 1, 1; 1, 1, 1, 1];
        end

    end
    [RewardFunc, ProbOfTransition] = ...
        restoreRnP2beforeGoal(RewardFunc, ProbOfTransition, QBeforeGoal, PBeforeGoal, FirstGoalPoint, GridSize);


    for iBl = 1 : NumOfBlocks -2
        for itrial = 1 : NumOfTrials
            StartPoint = PreyPredatorPos((iBl - 1) * NumOfTrials + itrial, 1);
            GoalPoint = PreyPredatorPos((iBl - 1) * NumOfTrials + itrial, 2);
            NumStep = 1;
            CurrState = StartPoint;
            ObsInCurrTrial = ObsInEnvMat(:, :, (iBl - 1) * NumOfTrials + itrial);
            % ================================= %

            while (CurrState ~= GoalPoint && NumStep <= MaxNumPerEps)
                [TempRewardFunc, TempTransFunc] = ...
                    RewardAtGoals(RewardFunc, ProbOfTransition, GoalPoint, GridSize);

                % === Action selection based on Softmax ===%
                [Q, Action] = ActionSel4Agent(CurrState, Temperature, Discount, NumOfAction, NumOfState,...
                    TempRewardFunc, TempTransFunc);
                if Action == 1
                    UpChoice(iBl, itrial, NumStep) = 1;
                end
                if Action == 2
                    RightChoice(iBl, itrial, NumStep) = 1;
                end
                if Action == 3
                    DownChoice(iBl,itrial, NumStep) = 1;
                end
                if Action == 4
                    LeftChoice(iBl,itrial, NumStep) = 1;
                end
                % ===== Update in the Environment =====%
                [NewState, Reward] = EnvModule(CurrState, GoalPoint, GridSize, NoUpAvail,...
                    NoDownAvail, ObsInCurrTrial, Action, PathAGreatLossAt, PathBGreatLossAt,...
                    PathCGreatLossAt, LossFreqInPathA, LossFreqInPathB, LossFreqInPathC,...
                    itrial, iBl, NumOfTrials, CheckAvailActInBl3, ObsKindVecIn3Block, TestPhaseNum);
                NumOfObser(CurrState, NewState, Action) = NumOfObser(CurrState, NewState, Action) + 1;
                ImRewardModel(NumStep, (iBl - 1) * NumOfTrials + itrial) = Reward;
                PreyPosAtEachAttempt(NumStep, (iBl - 1) * NumOfTrials + itrial) = CurrState;
                % ====================================== %

                % == Update the reward function and transition function == %
                if NewState == GoalPoint
                else
                    RewardFunc(CurrState, NewState, Action) = ...
                        (NumOfObser(CurrState, NewState, Action) * RewardFunc(CurrState, NewState, Action) + Reward)/(NumOfObser(CurrState, NewState, Action) + 1);
                end
                ProbOfTransition(CurrState, NewState, Action) = ...
                    ProbOfTransition(CurrState, NewState, Action) + LearningRate_T;
                ProbOfTransition(CurrState, :, Action) = ...
                    ProbOfTransition(CurrState, :, Action) ./ sum(ProbOfTransition(CurrState, :, Action));

                [RewardFunc, ProbOfTransition, ObservedCell] = ...
                    AdjustProb(RewardFunc, ProbOfTransition, ObservedCell, GridSize, rL);
                CurrState = NewState;
                NumStep = NumStep + 1;
            end
            PreyPosAtEachAttempt(NumStep, (iBl - 1) * NumOfTrials + itrial) = CurrState;
            NumStepVec((iBl -1 ) * NumOfTrials + itrial) = NumStep - 1;
        end
    end

    % =================================== %
    % In this section I test whether model based could replan or not
% % % % % %     for iBl = 8 : NumOfBlocks
% % % % % %         for itrial = 1 : NumOfTrials
% % % % % %             StartPoint = PreyPredatorPos((iBl - 1) * NumOfTrials + itrial, 1);
% % % % % %             GoalPoint = PreyPredatorPos((iBl - 1) * NumOfTrials + itrial, 2);
% % % % % %             NumStep = 1;
% % % % % %             CurrState = StartPoint;
% % % % % %             ObsInCurrTrial = ObsInEnvMat(:, :, (iBl - 1) * NumOfTrials + itrial);
% % % % % %             % ================================= %
% % % % % %             HaveVisited = 0;
% % % % % %             while (CurrState ~= GoalPoint && NumStep <= MaxNumPerEps)
% % % % % %                 [TempRewardFunc, TempTransFunc] = ...
% % % % % %                     RewardAtGoals(RewardFunc, ProbOfTransition, GoalPoint, GridSize);
% % % % % % 
% % % % % %                 if (ObsKindVecIn3Block((iBl - 8) * NumOfTrials + itrial) == 1)
% % % % % %                     if (HaveVisited == 1) || (CurrState == 8)
% % % % % %                         TempTransFunc(8, 12, 2) = 0;
% % % % % %                         HaveVisited = 1;
% % % % % %                     end
% % % % % %                 end
% % % % % %                 % === Action selection based on Softmax ===%
% % % % % %                 [Q, Action] = ActionSel4Agent(CurrState, Temperature, Discount, NumOfAction, NumOfState,...
% % % % % %                     TempRewardFunc, TempTransFunc);
% % % % % %                 if Action == 1
% % % % % %                     UpChoice(iBl, itrial, NumStep) = 1;
% % % % % %                 end
% % % % % %                 if Action == 2
% % % % % %                     RightChoice(iBl, itrial, NumStep) = 1;
% % % % % %                 end
% % % % % %                 if Action == 3
% % % % % %                     DownChoice(iBl,itrial, NumStep) = 1;
% % % % % %                 end
% % % % % %                 if Action == 4
% % % % % %                     LeftChoice(iBl,itrial, NumStep) = 1;
% % % % % %                 end
% % % % % %                 % ===== Update in the Environment =====%
% % % % % %                 [NewState, Reward] = EnvModule(CurrState, GoalPoint, GridSize, NoUpAvail,...
% % % % % %                     NoDownAvail, ObsInCurrTrial, Action, PathAGreatLossAt, PathBGreatLossAt,...
% % % % % %                     PathCGreatLossAt, LossFreqInPathA, LossFreqInPathB, LossFreqInPathC,...
% % % % % %                     itrial, iBl, NumOfTrials, CheckAvailActInBl3, ObsKindVecIn3Block, TestPhaseNum);
% % % % % %                 NumOfObser(CurrState, NewState, Action) = NumOfObser(CurrState, NewState, Action) + 1;
% % % % % %                 ImRewardModel(NumStep, (iBl - 1) * NumOfTrials + itrial) = Reward;
% % % % % %                 PreyPosAtEachAttempt(NumStep, (iBl - 1) * NumOfTrials + itrial) = CurrState;
% % % % % %                 % ====================================== %
% % % % % % 
% % % % % %                 % == Update the reward function and transition function == %
% % % % % %                 if NewState == GoalPoint
% % % % % %                 else
% % % % % %                     RewardFunc(CurrState, NewState, Action) = ...
% % % % % %                         (NumOfObser(CurrState, NewState, Action) * RewardFunc(CurrState, NewState, Action) + Reward)/(NumOfObser(CurrState, NewState, Action) + 1);
% % % % % %                 end
% % % % % %                 ProbOfTransition(CurrState, NewState, Action) = ...
% % % % % %                     ProbOfTransition(CurrState, NewState, Action) + LearningRate_T;
% % % % % %                 ProbOfTransition(CurrState, :, Action) = ...
% % % % % %                     ProbOfTransition(CurrState, :, Action) ./ sum(ProbOfTransition(CurrState, :, Action));
% % % % % %                 % ====================================== %
% % % % % %                 [RewardFunc, ProbOfTransition, ObservedCell] = ...
% % % % % %                     AdjustProbInRepMod(RewardFunc, ProbOfTransition, ObservedCell, GridSize, rL, CurrState);
% % % % % %                 CurrState = NewState;
% % % % % %                 NumStep = NumStep + 1;
% % % % % %             end
% % % % % %             PreyPosAtEachAttempt(NumStep, (iBl - 1) * NumOfTrials + itrial) = CurrState;
% % % % % %             NumStepVec((iBl -1 ) * NumOfTrials + itrial)= NumStep - 1;
% % % % % %         end
% % % % % %     end
    ID = num2str(iSim);
    SubjectID = strcat('Sb7', ID);
    save(strcat(Directory,'SimulatedData4MeshNFmin\',SubjectID,'.mat'), ...
        'ImRewardModel', 'UpChoice', 'RightChoice', 'DownChoice', 'LeftChoice', 'Q',...
        'PreyPosAtEachAttempt', 'PreyPredatorPos', 'ObsKindVecIn3Block', 'NumOfObser',...
        'LossFreqInPathA', 'LossFreqInPathB', 'LossFreqInPathC', 'ProbOfTransition', 'RewardFunc')
end







