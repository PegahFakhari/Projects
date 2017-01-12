function  [NewSt, CurrReward] = ...
    EnvModule(CurrState, Goal, GridSize, NoUpAvail,...
    NoDownAvail, ObsInCurrTrial, Action, PathAGreatLossAt, PathBGreatLossAt,...
    PathCGreatLossAt, GreatLossFreqInPathA, GreatLossFreqInPathB, GreatLossFreqInPathC,...
    TrialsCount, BlockCount, NumOfTrials, CheckAvailActInBl3, ObsKindVecIn3Block, TestPhaseNum)


[i,j] = ind2sub(GridSize,CurrState);

% check full obstacles
if Action == 1 % if action is up
    if (i - 1) ~= 0
        i = i - 1;
        NewSt = sub2ind(GridSize,i,j);
        CurrObs = ObsInCurrTrial(i,j);
        PosChanged = 1;
        if CurrObs == 1
            i = i + 1;
            NewSt = sub2ind(GridSize,i,j);
            PosChanged = 0; % because later NoUpAvail will be checked
        end
    else
        NewSt = CurrState;
    end
end

if Action == 2 % if action is right
    if (j + 1) ~= GridSize(2) + 1
        j = j + 1;
        NewSt = sub2ind(GridSize,i,j);
        CurrObs = ObsInCurrTrial(i,j);
        if CurrObs == 1
            j = j - 1;
            NewSt = sub2ind(GridSize,i,j);
        end
    else
        NewSt = CurrState;
    end
end

if Action == 3 % if action is down
    if (i + 1) ~= GridSize(1) + 1
        i = i + 1;
        NewSt = sub2ind(GridSize,i,j);
        CurrObs = ObsInCurrTrial(i,j);
        PosChanged = 1;
        if CurrObs == 1
            i = i - 1;
            NewSt = sub2ind(GridSize,i,j);
            PosChanged = 0; % because later NoDownAvail will be checked
        end
    else
        NewSt = CurrState;
    end
end

if Action == 4 % if action is left
    if (j - 1) ~= 0
        j = j - 1;
        NewSt = sub2ind(GridSize,i,j);
        CurrObs = ObsInCurrTrial(i,j);
        if CurrObs == 1
            j = j + 1;
            NewSt = sub2ind(GridSize,i,j);
        end
    else
        NewSt = CurrState;
    end
end
CurrReward = -1;

% Check the partial Obstacles
% NoUpAvail
if find(CurrState == NoUpAvail)
    if (Action == 1) && (PosChanged)
        i = i + 1;
        NewSt = sub2ind(GridSize,i,j);
    end
end
% NoDownAvail
if find(CurrState == NoDownAvail)
    if (Action == 3) && (PosChanged)
        i = i - 1;
        NewSt = sub2ind(GridSize,i,j);
    end
end
if BlockCount >= TestPhaseNum
    % obstacles: (#1) right into 12 from 8;
    % the rest is the same as Block1 & 2
    CurrentObsKind = ...
        ObsKindVecIn3Block((BlockCount - TestPhaseNum) * NumOfTrials +TrialsCount);
    if CurrentObsKind == 1
        if CurrState == CheckAvailActInBl3 % CurrState 8
            if NewSt == CurrState
            else
                if Action == 2 % if action was right
                    j = j - 1;
                    NewSt = sub2ind(GridSize,i,j);
                end
            end
        end
    end
end

if NewSt == Goal % check the goal
    CurrReward = 100;
else
    % check the two Obs in Path A
    if NewSt == PathAGreatLossAt(1)
        if CurrState ~= PathAGreatLossAt(1)  % In both directions
            CurrReward = GreatLossFreqInPathA(1, (BlockCount - 1) * NumOfTrials + TrialsCount);
        end
    end
    if NewSt == PathAGreatLossAt(2)
        if CurrState ~= PathAGreatLossAt(2)  % In both directions
            CurrReward = GreatLossFreqInPathA(2, (BlockCount - 1) * NumOfTrials + TrialsCount);
        end
    end
    % check the Obs in Path B
    if NewSt == PathBGreatLossAt
        if CurrState ~= PathBGreatLossAt % In both directions
            CurrReward = GreatLossFreqInPathB((BlockCount - 1) * NumOfTrials + TrialsCount);
        end
    end
    % check the two Obs in Path C
    if NewSt == PathCGreatLossAt(1)
        if CurrState ~= PathCGreatLossAt(1)  % In both directions
            CurrReward = GreatLossFreqInPathC(1, (BlockCount - 1) * NumOfTrials + TrialsCount);
        end
    end
    if NewSt == PathCGreatLossAt(2)
        if CurrState ~= PathCGreatLossAt(2)  % In both directions
            CurrReward = GreatLossFreqInPathC(2, (BlockCount - 1) * NumOfTrials + TrialsCount);
        end
    end
end







