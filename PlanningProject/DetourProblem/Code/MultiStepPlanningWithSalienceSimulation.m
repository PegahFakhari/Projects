% Reward (-1) and transition Based Multistep Planning model
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
Discount = 0.99;
Temperature = .4;
LearningRate_T = 0.75;
LearningRate_R = 0.16;
NumOfSimulation  = 1000;
SimulatedAveR = zeros(NumOfSimulation, NumOfBlocks * NumOfTrials);
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
    nOfSalientreward = zeros(NumOfState, NumOfState, NumOfAction);
    ObservedCell = zeros(NumOfState, NumOfAction);
    iSMultiStepUsed = zeros(MaxNumPerEps, NumOfTrials * NumOfBlocks);

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

    w = 1;
    checkme = 0;
    for iBl = 1 : NumOfBlocks -2
        for itrial = 1 : NumOfTrials
            StartPoint = PreyPredatorPos((iBl - 1) * NumOfTrials + itrial, 1);
            GoalPoint = PreyPredatorPos((iBl - 1) * NumOfTrials + itrial, 2);
            NumStep = 1;
            CurrState = StartPoint;
            ObsInCurrTrial = ObsInEnvMat(:, :, (iBl - 1) * NumOfTrials + itrial);

            RSaliance = 1; % this shows that reward is -1
            TSaliance = 1; % this shows that transition is possible
            % ================================= %

            while (CurrState ~= GoalPoint && NumStep <= MaxNumPerEps)
                [TempRewardFunc, TempTransFunc] = ...
                    RewardAtGoals(RewardFunc, ProbOfTransition, GoalPoint, GridSize);

                % === Find a path to Goal using Model based === %
                if (RSaliance == 1 || TSaliance == 1)
                    Path = PathFinder(CurrState, GoalPoint, Temperature, Discount, ...
                        NumOfAction, NumOfState, TempRewardFunc, TempTransFunc);
                    RSaliance = 0;
                    TSaliance = 0;
                    iSMultiStepUsed(NumStep, (iBl - 1) * NumOfTrials + itrial) = 1;
                    checkme = checkme + 1;
                end

                % ===== Check with the environment =====%
                Action = Path(2, (Path(1,:) == CurrState));
                [NewState, Reward] = EnvModule(CurrState, GoalPoint, GridSize, NoUpAvail,...
                    NoDownAvail, ObsInCurrTrial, Action, PathAGreatLossAt, PathBGreatLossAt,...
                    PathCGreatLossAt, LossFreqInPathA, LossFreqInPathB, LossFreqInPathC,...
                    itrial, iBl, NumOfTrials, CheckAvailActInBl3, ObsKindVecIn3Block, TestPhaseNum);
                NumOfObser(CurrState, NewState, Action) = NumOfObser(CurrState, NewState, Action) + 1;
                ImRewardModel(NumStep, (iBl - 1) * NumOfTrials + itrial) = Reward;
                % ====================================== %

                % ===== Check salience in Reward function ===== %
                if Reward ~= -1
                    if Reward == 100
                    else
                        nOfSalientreward(CurrState, NewState, Action) = ...
                            nOfSalientreward(CurrState, NewState, Action) +  1;
                        if nOfSalientreward(CurrState, NewState, Action) == 1
                            RSaliance = 1;
                        else
                            if nOfSalientreward(CurrState, NewState, Action)/NumOfObser(CurrState, NewState, Action) <0.5
                                RSaliance = 1;
                            end
                        end
                    end
                end

                % ==== Check salience in Transition function ==== %
                if NewState == CurrState
                    TSaliance = 1;
                end
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
                % ====================================== %

                % I check whether agent had chosen any action that was not available/ in other words hit the obstacle. this is easy
                % when you check the probability of staying in current state even after you select an action (and you're not in the border
                % of the grid). The result of the following function are two vectors and each
                % has two rows; first row is from other cells to Goal; the
                % second row is from goal to other cells
                [RewardFunc, ProbOfTransition, ObservedCell] = ...
                    AdjustProb(RewardFunc, ProbOfTransition, ObservedCell, GridSize, rL);
                CurrState = NewState;
                NumStep = NumStep + 1;
            end
            NumStepVec(w)= NumStep - 1;
            w = w + 1;
        end
    end

    % =================================== %
    % =================================== %


    % In this section I test whether model based could replan or not
    % after it estimates R and T functions in
    % "ReplaninngPredictionWLearingModelBased" mfile

    NumStepVec = zeros(1, 2 * NumOfTrials + 1);
    CheckPlanning = 3.14 * ones(1, 2 * NumOfTrials + 1);
    CheckRePlanning = 3.14 * ones(1, 2 * NumOfTrials + 1);
    % ================================= %
    OptimalPathInPlanning = [3, 7, 8, 12, 16, 20, 19, 23, 27];
    OptimalPathInRePlanning = [3, 7, 8, 7, 6, 5, 9, 13, 14, 15, 19, 23, 27];
    PathInPlanning = zeros(1, numel(OptimalPathInPlanning));
    PathInPlanning(1) = 3;
    PathInRePlanning = zeros(1, numel(OptimalPathInRePlanning));
    PathInRePlanning(1) = 3;
    TempRewardFunc = RewardFunc;
    TempTransFunc = ProbOfTransition;

    % Test Re-planning w/ predetermined Transition and Reward functions %
    w = 1;
    for iBl = 8 : NumOfBlocks
        for itrial = 1 : NumOfTrials
            StartPoint = PreyPredatorPos((iBl - 1) * NumOfTrials + itrial, 1);
            GoalPoint = PreyPredatorPos((iBl - 1) * NumOfTrials + itrial, 2);
            NumStep = 1;
            CurrState = StartPoint;
            ObsInCurrTrial = ObsInEnvMat(:, :, (iBl - 1) * NumOfTrials + itrial);

            RSaliance = 0; % this shows that reward is -1
            TSaliance = 1; % this shows that transition is possible
            % ================================= %
            HaveVisited = 0;
            u = 2;
            while (CurrState ~= GoalPoint && NumStep <= MaxNumPerEps)
                [TempRewardFunc, TempTransFunc] = ...
                    RewardAtGoals(RewardFunc, ProbOfTransition, GoalPoint, GridSize);

                if (ObsKindVecIn3Block((iBl - 8) * NumOfTrials + itrial) == 1)
                    if (HaveVisited == 1) || (CurrState == 8)
                        TempTransFunc(8, 12, 2) = 0;
                        HaveVisited = 1;
                        TSaliance = 1;
                    end
                end

                % === Find a path to Goal using Model based === %
                if TSaliance == 1
                    if (TSaliance == 1 && TempTransFunc(8, 12, 2) == 0 && NumStep > 3)
                        clc
                    else
                        Path = PathFinder(CurrState, GoalPoint, Temperature, Discount, ...
                            NumOfAction, NumOfState, TempRewardFunc, TempTransFunc);
                        TSaliance = 0;
                        iSMultiStepUsed(NumStep, (iBl - 1) * NumOfTrials + itrial) = 1;
                    end
                end

                % ===== Update in the Environment =====%
                Action = Path(2, (Path(1,:) == CurrState));
                [NewState, Reward] = EnvModule(CurrState, GoalPoint, GridSize, NoUpAvail,...
                    NoDownAvail, ObsInCurrTrial, Action, PathAGreatLossAt, PathBGreatLossAt,...
                    PathCGreatLossAt, LossFreqInPathA, LossFreqInPathB, LossFreqInPathC,...
                    itrial, iBl, NumOfTrials, CheckAvailActInBl3, ObsKindVecIn3Block, TestPhaseNum);
                NumOfObser(CurrState, NewState, Action) = NumOfObser(CurrState, NewState, Action) + 1;
                ImRewardModel(NumStep, (iBl - 1) * NumOfTrials + itrial) = Reward;
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
                % ====================================== %
                [RewardFunc, ProbOfTransition, ObservedCell] = ...
                    AdjustProbInRepMod(RewardFunc, ProbOfTransition, ObservedCell, GridSize, rL, CurrState);
                CurrState = NewState;
                if ObsKindVecIn3Block((iBl - 8) * NumOfTrials + itrial) ~= 1
                    PathInPlanning(u) = CurrState;
                else
                    PathInRePlanning(u) = CurrState;
                end
                NumStep = NumStep + 1;
                u = u + 1;
            end
            if ObsKindVecIn3Block((iBl - 8) * NumOfTrials + itrial) ~= 1
                if numel(OptimalPathInPlanning) == numel(PathInPlanning)
                    CheckPlanning(w) = sum(OptimalPathInPlanning - PathInPlanning);
                end
            else
                if numel(OptimalPathInRePlanning) == numel(PathInRePlanning)
                    CheckRePlanning(w) = sum(OptimalPathInRePlanning - PathInRePlanning);
                end
            end
            NumStepVec(w)= NumStep - 1;
            w = w + 1;
        end
    end

    % =================================== %
    % =================================== %

    % == Compare the total rewards Model and subject earned == %
    % ============ Moving Average for Reward ============ %
    RewardPerTrial4Model = sum(ImRewardModel);
    Window = 5;
    AveRModel = zeros(1, numel(RewardPerTrial4Model) - Window);
    for c = 1 : numel(RewardPerTrial4Model) - Window
        AveRModel(c) = mean(RewardPerTrial4Model(c : c + Window));
    end
    SimulatedAveR(iSim, 1: numel(AveRModel)) = AveRModel;
end



rectangle('Position', [0,0, 120,100], 'FaceColor', [.95 .87 .73], 'EdgeColor', [.95 .87 .73])
rectangle('Position', [120,0, 20,100], 'FaceColor', [.91 .91 .91], 'EdgeColor', [.91 .91 .91])
rectangle('Position', [140,0, 40,100], 'FaceColor', [.96 .92 .92], 'EdgeColor', [.96 .92 .92])
hold on
plot(SimulatedAveR(:, 1: 175)', 'color', 140/255 *[1 1 1])
hold on
plot(mean(SimulatedAveR(:, 1: 175)), 'r')
xlim([0 175])
ylim([0 100])

















