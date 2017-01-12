clc
clear
close all
NumOfBlocks = 10; % two of them are for training and one for test
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
rW = -1;
rL = -1;
rG = 100;
r = -1;
RewStr = [rW, rG, r, rL];
Ifwhilenotmet = 200;
NumOfSubj  = 32; % 19 for Exp 1, 36 for Exp2, 32 for Exp 3
Decay = 0;
load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Model Fit Result\ModelBasedLastR\ModelBasedLastRExp3'))
% fminresult
SubjLLInTest = zeros(1, NumOfSubj);
pmatOfallSubj = zeros(4, NumOfSubj);
SelPair = [3, 27];
AtCell = 7;
% ================================= %
for iSub = 1 : NumOfSubj
    clc
    BeInState = 0;
    GoU = 0;
    GoR = 0;
    GoD = 0;
    GoL = 0;
    fprintf('sim=%d', iSub, NumOfSubj);
    ID = num2str(iSub);
%     SubjectID = strcat('Exp1S', ID); % Exp1
    %     SubjectID = strcat('mdfs', ID); % Exp2
        SubjectID = strcat('STEN', ID, '-Whole'); % Exp3
    load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\SimulatedData4MeshNFmin\Exp 3\', SubjectID,'.mat'))
    Discount = 1/(1 + exp(fminresult(iSub, 1)));
    Temperature = fminresult(iSub, 2);
    PrB = 0;
    rL = RewStr(4);
    NumOfObser = zeros(NumOfState, NumOfState, NumOfAction);
    ObservedCell = zeros(NumOfState, NumOfAction);
    % ==== Reward and Transition Function ==== %
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

    % ================================= %
    for iBl = 1 : NumOfBlocks
        for itrial = 1 : NumOfTrials
            NumStep = 1;
            GoalPoint = PreyPredatorPos((iBl -1 ) * NumOfTrials + itrial, 2);
            CurrState = PreyPosAtEachAttempt(NumStep, (iBl -1 ) * NumOfTrials + itrial);
            % ================================ %
            % Check the obstacle in test phase at cell 8 %
            HaveVisited = 0;
            % ================================ %
            while (CurrState ~= GoalPoint && NumStep <= MaxNumPerEps)
                [TempRewardFunc, TempTransFunc] = ...
                    RewardAtGoals(RewardFunc, ProbOfTransition, GoalPoint, GridSize);

                if UpChoice(iBl,itrial, NumStep) == 1
                    Action = 1;
                end
                if RightChoice(iBl,itrial, NumStep) == 1
                    Action = 2;
                end
                if DownChoice(iBl,itrial, NumStep) == 1
                    Action = 3;
                end
                if LeftChoice(iBl,itrial, NumStep) == 1
                    Action = 4;
                end

                NewState = PreyPosAtEachAttempt(NumStep + 1, (iBl -1 ) * NumOfTrials + itrial);
                NumOfObser(CurrState, NewState, Action) = NumOfObser(CurrState, NewState, Action) + 1;
                Reward = ImRewardData(NumStep, (iBl - 1) * NumOfTrials + itrial);
                % ====================================== %
                % == Update the reward function and transition function == %
                if Reward == 100
                else
                    RewardFunc(CurrState, NewState, Action) = Reward;
                end
                ProbOfTransition(CurrState, NewState, Action) = ...
                    ProbOfTransition(CurrState, NewState, Action) + LearningRate_T;
                ProbOfTransition(CurrState, :, Action) = ...
                    ProbOfTransition(CurrState, :, Action) ./ sum(ProbOfTransition(CurrState, :, Action));

                [RewardFunc, ProbOfTransition, ObservedCell] = ...
                    AdjustProb(RewardFunc, ProbOfTransition, ObservedCell, GridSize, rL);
                % ====================================== %
                if iBl > 7
                    % Q-value iteration needs to have the most updated T
                    if (ObsKindVecIn3Block((iBl - 8) * NumOfTrials + itrial) == 1)
                        if (HaveVisited == 1) || (CurrState == 8)
                            TempTransFunc(8, 12, 2) = 0;
                            HaveVisited = 1;
                        end
                    end

                    % ====================================== %
                    q = [0 0 0 0];
                    q = repmat(q, NumOfState, 1);
                    ThetaConvergence = 0.002;
                    DeltaConvergence = 1;
                    NumInwhileLoop = 0;
                    while (DeltaConvergence > ThetaConvergence) && (NumInwhileLoop < Ifwhilenotmet)
                        maxq = max(q, [], 2);
                        maxq = repmat(maxq', [NumOfState, 1, NumOfAction]);
                        Q = sum(TempTransFunc .* (TempRewardFunc + Discount * maxq), 2);
                        Q = reshape(Q, [NumOfState, NumOfAction]);
                        DeltaConvergence = max(0, sum(sum(abs(q - Q))));
                        q = Q;
                        NumInwhileLoop = NumInwhileLoop + 1;
                    end
                    NewQ = Q(CurrState, :);
                    ProbAct = exp(NewQ(Action) / Temperature) / sum(exp(NewQ / Temperature));

                    if  ProbAct < 10e-300
                        PrB = PrB - 800;
                    else
                        PrB = PrB + log( ProbAct );
                    end
                    if PreyPredatorPos((iBl -1 ) * NumOfTrials + itrial, 1) == SelPair(1) && GoalPoint == SelPair(2)
                        if (ObsKindVecIn3Block((iBl - 8) * NumOfTrials + itrial) ~= 1)
                            if CurrState == AtCell
                                BeInState = BeInState + 1;
                                if DownChoice(iBl, itrial, NumStep) == 1
                                    GoD = GoD + 1;
                                end
                                if LeftChoice(iBl, itrial, NumStep) == 1
                                    GoL = GoL + 1;
                                end
                                if RightChoice(iBl, itrial, NumStep) == 1
                                    GoR = GoR + 1;
                                end
                                if UpChoice(iBl, itrial, NumStep) == 1
                                    GoU = GoU + 1;
                                end
                            end
                        end
                    end
                end
                % ====================================== %
                CurrState = NewState;
                NumStep = NumStep + 1;
            end
        end
    end
    pmatOfallSubj(1, iSub) = GoU/BeInState;
    pmatOfallSubj(2, iSub) = GoR/BeInState;
    pmatOfallSubj(3, iSub) = GoD/BeInState;
    pmatOfallSubj(4, iSub) = GoL/BeInState;
    SubjLLInTest(iSub) = -PrB;
end

