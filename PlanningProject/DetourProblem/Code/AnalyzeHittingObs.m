% Analyzing the rt in Re-planning Exp
clc
clear
close all
NumOfBlocks = 10; % two of them are for training and one for test
BlockNumInTest = 8;
PrestestBlock = 7;
NumOfTrials = 20;
MaxNumPerEps = 15;
GoalReward = 100;
GridSize = [4, 7];
NumOfState = prod(GridSize);
NumOfAction = 4;
NumOfSubj = 32;
hit75 = 9;
RateOfHit = zeros(NumOfSubj * NumOfBlocks * NumOfTrials, 3);
RateOfHitG = zeros(NumOfSubj, 2);
RateOfHitObs = zeros(NumOfSubj * NumOfBlocks * NumOfTrials, 3);
HitObs = zeros(NumOfSubj * (NumOfBlocks - (PrestestBlock - 1)) * NumOfTrials, 4);
AveRewpSubpBlock = zeros(NumOfSubj * 2, 3);
AveRewpSubInTest = zeros(NumOfSubj * 3, 3);

VH = [5, 13; 7, 13];
H = [3, 17; 5, 22; 5, 15];
E = [3, 27; 7, 23];
VE = [7, 16; 6, 20; 3, 14];
AllTestPairs = [VH; H; E; VE];
% == Optimal path if the Path is NOT blocked == %
OptimalPathInTestPhase = zeros(size(AllTestPairs, 1), MaxNumPerEps);
OptimalPathInTestPhase(1,:) = [5, 6, 7, 8, 12, 16, 20, 19, 15, 14, 13, 0, 0, 0, 0];
OptimalPathInTestPhase(2,:) = [7, 8, 12, 16, 20, 19, 15, 14, 13, 0, 0, 0, 0, 0, 0];
OptimalPathInTestPhase(3,:) = [3, 7, 8, 12, 16, 20, 19, 23, 22, 21, 17, 0, 0, 0, 0];
OptimalPathInTestPhase(4,:) = [5, 6, 7, 8, 12, 16, 20, 19, 23, 22, 0, 0, 0, 0, 0];
OptimalPathInTestPhase(5,:) = [5, 6, 7, 8, 12, 16, 20, 19, 15, 0, 0, 0, 0, 0, 0];
OptimalPathInTestPhase(6,:) = [3, 7, 8, 12, 16, 20, 19, 23, 27, 0, 0, 0, 0, 0, 0];
OptimalPathInTestPhase(7,:) = [7, 8, 12, 16, 20, 19, 23, 0, 0, 0, 0, 0, 0, 0, 0];
OptimalPathInTestPhase(8,:) = [7, 8, 12, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
OptimalPathInTestPhase(9,:) = [6, 7, 8, 12, 16, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0];
OptimalPathInTestPhase(10,:) = [3, 7, 8, 12, 16, 20, 19, 15, 14, 0, 0, 0, 0, 0, 0];

% == Optimal path if the Path is blocked == %
OptimalBPathInTestPhase = zeros(size(AllTestPairs, 1), MaxNumPerEps);
OptimalBPathInTestPhase(1,:) = [5, 6, 7, 8, 7, 6, 5, 9, 13, 0, 0, 0, 0, 0, 0];
OptimalBPathInTestPhase(2,:) = [7, 8, 7, 6, 5, 9, 13, 0, 0, 0, 0, 0, 0, 0, 0];
OptimalBPathInTestPhase(3,:) = [3, 7, 8, 7, 6, 5, 9, 13, 17, 0, 0, 0, 0, 0, 0];
OptimalBPathInTestPhase(4,:) = [5, 6, 7, 8, 7, 6, 5, 9, 13, 14, 15, 19, 23, 22, 0];
OptimalBPathInTestPhase(5,:) = [5, 6, 7, 8, 7, 6, 5, 9, 13, 14, 15, 0, 0, 0, 0];
OptimalBPathInTestPhase(6,:) = [3, 7, 8, 7, 6, 5, 9, 13, 14, 15, 19, 23, 27, 0, 0];
OptimalBPathInTestPhase(7,:) = [7, 8, 7, 6, 5, 9, 13, 14, 15, 19, 23, 0, 0, 0, 0];
OptimalBPathInTestPhase(8,:) = [7, 8, 7, 6, 5, 9, 13, 14, 15, 19, 20, 16, 0, 0, 0];
OptimalBPathInTestPhase(9,:) = [6, 7, 8, 7, 6, 5, 9, 13, 14, 15, 19, 20, 0, 0, 0];
OptimalBPathInTestPhase(10,:) = [3, 7, 8, 7, 6, 5, 9, 13, 14, 0, 0, 0, 0, 0, 0];

for iSub = 1 : NumOfSubj
    clc
    fprintf('sim=%d', iSub);
    
    ID = num2str(iSub);
    SubjectID = strcat('STEN', ID,'-Whole');
    load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Data\Paid Subjects With 10 Blocks\', SubjectID,'.mat'))
    
    % ============================================== %
    % === First check the 5 trials at the end of learning phase === %
    PreyPosIn5TrialsBeforeTest = ...
        PreyPosAtEachAttempt(:, (PrestestBlock - 1) * NumOfTrials - 4 : (PrestestBlock - 1) * NumOfTrials);
    
    OptimalPathSeq = [3, 7, 8, 12, 16, 20, 19, 23];
    LearningInTrial = zeros(1, 5);
    % In this loop I assumed that subjects didn't hit the boundaries and
    % obstacles
    for i = 1 : 5
        IndCheck = 0;
        CheckThisTrial = PreyPosIn5TrialsBeforeTest(:, i);
        for j = 1 : numel(OptimalPathSeq)
            if find(OptimalPathSeq(j) ~= CheckThisTrial(j))
                IndCheck = IndCheck + 1;
            end
        end
        if IndCheck == 0
            LearningInTrial(i) = 1;
        end
    end
    
    % ============================================== %
    % === Second check subjects choose path C1A2 after path B === %
    % From those which were obstructed, how many times subject
    % chose to select path C1A2 and not other paths
    PreyPosInTestPhase = ...
        PreyPosAtEachAttempt(:, (NumOfBlocks - 3) * NumOfTrials + 1: NumOfBlocks * NumOfTrials);
    
    PlanOrReplan = zeros(4, size(PreyPosInTestPhase, 2));
    PlanOrReplan(4, :) = ObsKindVecIn3Block;
    
    for i = 1 : size(PreyPosInTestPhase, 2)
        SG = PreyPredatorPos((BlockNumInTest - 1) * NumOfTrials + i, :);
        Trace = PreyPosInTestPhase(:, i);
        rewardeTrace = ImRewardData(:, (BlockNumInTest - 1) * NumOfTrials + i);
        Ind = find(Trace == 0);
        if Ind
            Trace(Ind(1)) = SG(2);
            Trace(Trace == 0) = [];
        else
            rewardeTrace(rewardeTrace == 0) = [];
            if rewardeTrace(end) == 100
                Trace(Trace == 0) = [];
                ExtendedTrace = [Trace; SG(2)];
                Trace = ExtendedTrace;
            end
        end
        
        % If the optimal path is blocked
        if ObsKindVecIn3Block(i) == 1
            if Trace(end) == SG(2)
                if mod(i, NumOfTrials) == 0
                    SecInd = 20;
                else
                    SecInd = mod(i, NumOfTrials);
                end
                if (i == 20) || (i == 40) || (i == 60)
                    FirstInd = floor(i/NumOfTrials) - 1 ;
                else
                    FirstInd = floor(i/NumOfTrials);
                end
                downcheck = DownChoice(BlockNumInTest + FirstInd, SecInd, :);
                CheckFlagg = 0;
                fg = 1;
                while CheckFlagg == 0
                    CurrOptt = OptimalBPathInTestPhase(fg, :);
                    CurrOptt(CurrOptt == 0) = [];
                    if Trace(1) == CurrOptt(1)
                        if Trace(end) == CurrOptt(end)
                            CheckFlagg = 1;
                            K2 = 0;
                            for ds = 1 : numel(CurrOptt)
                                if CurrOptt(ds) == 8
                                    if find(7 == Trace)
                                        K2 = K2 + 1;
                                    end
                                else
                                    if find(CurrOptt(ds) == Trace)
                                        K2 = K2 + 1;
                                    end
                                end
                            end
                            Ind8 = find(CurrOptt == 8) - 1;
                            
                            if K2 == numel(CurrOptt)
                                PlanOrReplan(2, i) = 1; % Replann correclty after checking the optimal path
                                PlanOrReplan(3, i) = 1; % check the optimal path but not replan correclty
                                if find(17 == Trace)
                                    if 17 == Trace(end)
                                    else
                                        PlanOrReplan(2, i) = 0;
                                    end
                                end
                            end
                            if downcheck(Ind8) && (Trace(Ind8) == 7)
                                PlanOrReplan(3, i) = 1;
                            end
                            
                        end
                    end
                    fg = fg + 1;
                end
                
            end
        else
            % If the optimal path is NOT blocked
            if Trace(end) == SG(2)
                % Which row of the OptimalPathInTestPhase is related to the current trace
                CheckFlagg = 0;
                fg = 1;
                while CheckFlagg == 0
                    CurrOptt = OptimalPathInTestPhase(fg, :);
                    CurrOptt(CurrOptt == 0) = [];
                    if Trace(1) == CurrOptt(1)
                        if Trace(end) == CurrOptt(end)
                            CheckFlagg = 1;
                            K3 = 0;
                            for ds = 1 : numel(CurrOptt)
                                if find(CurrOptt(ds) == Trace)
                                    K3 = K3 + 1;
                                end
                            end
                            if K3 == numel(CurrOptt)
                                PlanOrReplan(1, i) = 1; % no need for replanning because the optimal path is open
                            end
                        end
                    end
                    fg = fg + 1;
                end
            end
        end
        
    end
    
    % PlanOrReplan: first row, plan optimally; second row, replan optimally;
    % third row, check the accident but not replanned optimally
    % forth row shows the blocks in test phase
    
    % =========================================== %
    % =========== Learning in Pre-test Block ============ %
    PretestPositions =...
        PreyPosAtEachAttempt(:, (PrestestBlock - 1) * NumOfTrials + 1 : PrestestBlock * NumOfTrials);
    PreyPredatorPosInPretest =...
        PreyPredatorPos((PrestestBlock - 1) * NumOfTrials + 1 : PrestestBlock * NumOfTrials, :);
    
    % Add goal at the end of the path %
    for  i = 1 : size(PretestPositions, 2)
        Goal = PreyPredatorPosInPretest(i,2);
        CheckThisTrial = PretestPositions(:, i);
        Ind = find(CheckThisTrial == 0);
        if Ind
            CheckThisTrial(Ind(1)) = Goal;
            PretestPositions(:, i) = CheckThisTrial;
        end
    end
    
    PretestInitG1 = [12, 15; 5, 20; 7, 13; 3, 21; 7, 23; 5, 27; 13, 5; 7, 15];
    PretestInitG2 = [14, 22; 13, 21];
    OptInG1 = 16;
    OptInG2 = 19;
    OptSelInPretest = zeros(1, size(PretestPositions, 2));
    G1G2 = zeros(1, size(PretestPositions, 2));
    G1 = 0;
    G2 = 0;
    
    
    for i = 1 : size(PretestPositions, 2)
        OptG1 = 0;
        OptG2 = 0;
        CheckThisTrial = PretestPositions(:, i);
        
        % check whether it's in group 1 or group 2
        currPos = PreyPredatorPosInPretest(i,:);
        for k = 1 : size(PretestInitG1,1)
            if currPos(1) == PretestInitG1(k, 1)
                if currPos(2) == PretestInitG1(k, 2)
                    G1 = 1;
                    G1G2(i) = 1;
                end
            end
        end
        
        for k = 1 : size(PretestInitG2,1)
            if currPos(1) == PretestInitG2(k, 1)
                if currPos(2) == PretestInitG2(k, 2)
                    G2 = 1;
                    G1G2(i) = 2;
                end
            end
        end
        
        if G1
            if find(OptInG1 == CheckThisTrial)
                OptG1 = OptG1 + 1;
            end
        end
        
        if G2
            if find(OptInG2 == CheckThisTrial)
                OptG2 = OptG2 + 1;
            end
        end
        
        if OptG1
            OptSelInPretest(i) = 1;
        end
        
        if OptG2
            OptSelInPretest(i) = 1;
        end
    end
    
    % ============================================== %
    % Testing the NULL hypothesis in optimal planning and replanning
    % sig min criteria: 10 for 13; 15 for 20 ; 26 out of 40
    
    nullHProInOptPlan = cumsum(binopdf((sum(PlanOrReplan(1, ObsKindVecIn3Block ~= 1))): ...
        1: (numel(ObsKindVecIn3Block) - numel(find(ObsKindVecIn3Block == 1))), ...
        (numel(ObsKindVecIn3Block) - numel(find(ObsKindVecIn3Block == 1))), .5));
    
    
    nullHProInOptReplan = cumsum(binopdf((sum(PlanOrReplan(2, ObsKindVecIn3Block == 1))): 1: ...
        numel(find(ObsKindVecIn3Block == 1)), numel(find(ObsKindVecIn3Block == 1)), .5));
    
    % This explains how many of not optimal replanning is related to not
    % checking the optimal path first and how many of them are related to not
    % all checking the optimal path; for instance if the total notoptimal
    % replanning 16 and then the nonoptreplanning is 17; it means that 1 of the
    % not optimal replanning is because of not find the second optimal and the
    % other 3 show that the subject didn't check the optimal plan at first!
    nullHProInNonOptReplan = cumsum(binopdf((sum(PlanOrReplan(3,ObsKindVecIn3Block == 1))): 1: ...
        numel(find(ObsKindVecIn3Block == 1)), numel(find(ObsKindVecIn3Block == 1)), .5));
    
    % === Testing the NULL hypothesis in learning in Pretest === %
    NullHProInPretestLearning = cumsum(binopdf(sum(OptSelInPretest): 1: numel(OptSelInPretest), numel(OptSelInPretest), .5));
    
    % =========================================== %
    % ============ Hit -75 in learning phase ============ %
    NumOfHit75inLearning = 0;
    % PrestestBlock for general purpuses.
    for hO = 1 : (PrestestBlock) * NumOfTrials
        Goal = PreyPredatorPos(hO,2);
        CheckThisTrial = PreyPosAtEachAttempt(:, hO);
        if find(hit75 == CheckThisTrial)
            NumOfHit75inLearning = NumOfHit75inLearning + 1;
            RateOfHit((iSub - 1) * NumOfTrials * NumOfBlocks + hO, 2) = 1; % Hit
        end
        RateOfHit((iSub - 1) * NumOfTrials * NumOfBlocks + hO, 3) = 0; % block
        %% actual block:
        % ceil(hO/NumOfTrials)
        
        RateOfHit((iSub - 1) * NumOfTrials * NumOfBlocks + hO, 1) = iSub; % subject
        
        % whether subject hits an obstacle/stay in current position
        CheckThisTrial(CheckThisTrial == 0) = [];
        RateOfHitObs((iSub - 1) * NumOfTrials * NumOfBlocks + hO, 1) = iSub;
        RateOfHitObs((iSub - 1) * NumOfTrials * NumOfBlocks + hO, 2) = ceil(hO/NumOfTrials);
        RateOfHitObs((iSub - 1) * NumOfTrials * NumOfBlocks + hO, 3) = numel(CheckThisTrial) - numel(unique(CheckThisTrial));
    end
    
    % ============ Hit -75 in Test phase ============ %
    NumOfHit75inTest = 0;
    % PrestestBlock if line 300
    for hO1 = ((PrestestBlock) * NumOfTrials) + 1 : NumOfBlocks * NumOfTrials
        Goal = PreyPredatorPos(hO1,2);
        CheckThisTrial = PreyPosAtEachAttempt(:, hO1);
        if find(hit75 == CheckThisTrial)
            NumOfHit75inTest = NumOfHit75inTest + 1;
            RateOfHit((iSub - 1) * NumOfTrials * NumOfBlocks + hO1, 2) = 1; % Hit
        end
        RateOfHit((iSub - 1) * NumOfTrials * NumOfBlocks + hO1, 3) = 1; % blocks
        %% Actual Block:
        % ceil(hO1/NumOfTrials)
        RateOfHit((iSub - 1) * NumOfTrials * NumOfBlocks + hO1, 1) = iSub; % subject
        
        % whether subject hits an obstacle/stay in current position
        CheckThisTrial(CheckThisTrial == 0) = [];
        RateOfHitObs((iSub - 1) * NumOfTrials * NumOfBlocks + hO1, 1) = iSub;
        RateOfHitObs((iSub - 1) * NumOfTrials * NumOfBlocks + hO1, 2) = ceil(hO1/NumOfTrials);
        RateOfHitObs((iSub - 1) * NumOfTrials * NumOfBlocks + hO1, 3) = numel(CheckThisTrial) - numel(unique(CheckThisTrial));
        
        % Save the pretest and the test block separately %
        % now I'm comparing Pretest and test for hitting obstacles and thus I have (PrestestBlock - 1)
        % but for hitting -75 or generally the rate of hitting obstacles I
        % should change (PrestestBlock - 1) to PrestestBlock
        HitObs((iSub - 1) * NumOfTrials * (NumOfBlocks - (PrestestBlock - 1)) + (hO1 - (PrestestBlock -1)* NumOfTrials), 1) = iSub;
        if ceil(hO1/NumOfTrials) == PrestestBlock
            HitObs((iSub - 1) * NumOfTrials * (NumOfBlocks - (PrestestBlock - 1)) + (hO1 - (PrestestBlock -1) * NumOfTrials), 3) = 0;
        else
            HitObs((iSub - 1) * NumOfTrials * (NumOfBlocks - (PrestestBlock - 1)) + (hO1 - (PrestestBlock -1) * NumOfTrials), 3) = 1;
        end
        HitObs((iSub - 1) * NumOfTrials * (NumOfBlocks - (PrestestBlock - 1)) + (hO1 - (PrestestBlock -1) * NumOfTrials), 2) = numel(CheckThisTrial) - numel(unique(CheckThisTrial));
        if hO1 > PrestestBlock * NumOfTrials
            if ObsKindVecIn3Block((hO1 - (PrestestBlock) * NumOfTrials)) == 1
                HitObs((iSub - 1) * NumOfTrials * (NumOfBlocks - (PrestestBlock - 1)) + (hO1 - (PrestestBlock -1) * NumOfTrials), 4) = 1;
            else
                HitObs((iSub - 1) * NumOfTrials * (NumOfBlocks - (PrestestBlock - 1)) + (hO1 - (PrestestBlock -1) * NumOfTrials), 4) = 0;
            end
        end
    end
    
    %     RateOfHit(iSub, 1) = NumOfHit75inLearning/ ((PrestestBlock - 1) * NumOfTrials);
    %     RateOfHit(iSub, 2) = NumOfHit75inTest/ (NumOfBlocks * NumOfTrials - (PrestestBlock * NumOfTrials));
    RateOfHitG(iSub, 1) = NumOfHit75inLearning;
    RateOfHitG(iSub, 2) = NumOfHit75inTest;
    
    % analyzing the reward at block 1 and 7:
    for kh = 1 : NumOfBlocks
        if kh == 1 || kh == PrestestBlock
            if kh == 1
                indez = 1;
            else
                indez = 2;
            end
            AveRewpSubpBlock( (iSub -1) * 2 + indez, 1) = iSub;
            AveRewpSubpBlock( (iSub -1) * 2 + indez, 2) = sum(sum((ImRewardData(:, (kh-1) * NumOfTrials + 1: (kh) * NumOfTrials))))/NumOfTrials;
            AveRewpSubpBlock( (iSub -1) * 2 + indez, 3) = kh;
        end
    end
    
    % analyzing the reward at block 8, 9, 10:
    for kh = 1 : NumOfBlocks
        if kh == 8 || kh == 9 || kh == 10
                switch kh
                    case 8
                        indez = 1;
                    case 9
                        indez = 2;
                    case 10      
                        indez = 3;
                end
            AveRewpSubInTest( (iSub -1) * 3 + indez, 1) = iSub;
            AveRewpSubInTest( (iSub -1) * 3 + indez, 2) = sum(sum((ImRewardData(:, (kh-1) * NumOfTrials + 1: (kh) * NumOfTrials))))/NumOfTrials;
            AveRewpSubInTest( (iSub -1) * 3 + indez, 3) = kh;
        end
    end
end

%===determine the directory===
a= 'FitWithMLE\AnalyzeHittingObs.m';
Directory = which(a);
Directory = Directory(1:(end-numel(a)));
%===determine the directory===

Result = 'Hit75Block';
% Result = 'HitObs';
% Result = 'Reward';
% Result = 'HitObsPerBlock';

header = {'Subject', 'Hit', 'blockNo'};
% header = {'Subject', 'Hit', 'blockNo', 'blockage'};
% header = {'Subject', 'Reward', 'blockNo'};
% header = {'Subject', 'blockNo', 'Obs'};

% data = num2cell(AveRewpSubpBlock);
% data = num2cell(AveRewpSubInTest);
% data = num2cell(RateOfHitObs);
data = num2cell(RateOfHit);

xlswrite(strcat(Directory, 'FitWithMLE\', Result,'.xls'),[header;data]);



