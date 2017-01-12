% Analyzing the rt in Re-planning Exp
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
Pairs = [12, 13; 20, 13; 12, 21; 3, 27; 6, 20; 5, 27; 3, 21; 6, 22;...
    12, 14; 15, 5; 13, 7; 14, 22; 21, 15; 23, 13; 20, 14; 14, 3; 6, 14;...
    15, 8; 11, 12; 15, 16; 3, 12; 3, 20; 3, 13; 12, 5; 23, 7; 21, 5;...
    19, 20; 20, 22; 15, 14; 13, 21; 15, 23; 20, 8; 15, 7; 13, 5];
PreTestPairs = [14, 22; 12, 15; 5, 20; 7, 13; 3, 21; 7, 23; 7, 15; 5, 13; 13, 21; 27, 5];
TestPairs = [5, 13; 7, 13; 3, 17; 5, 22; 5, 15; 3, 27; 7, 23; 7, 16; 6, 20; 3, 14];

%====================================%
% For all subject, check specific pair and see whether on average the rt
% for planning in critical cells is greater than the rt in regular cells.
NumOfSubj = 32;
TempPair = [7, 13];
gridtimeMat = zeros(NumOfState, NumOfSubj);
orderOfexp = zeros(5, NumOfSubj);
traceMat = zeros(5, MaxNumPerEps + 1, NumOfSubj); % 3, 21 repeats 5 times

for iSub = 1 : NumOfSubj
    clc
    fprintf('sim=%d', iSub);

    % Clear Memory
    PreyPredatorPos = 0;
    PreyPosAtEachAttempt = 0;
    beenInCell = zeros(1, NumOfState);

    ID = num2str(iSub);
    SubjectID = strcat('STEN', ID,'-Whole');
    load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Data\Paid Subjects With 10 Blocks\', SubjectID,'.mat'))

    q1 = 0;
    for h = 1 : 180 % size(PreyPredatorPos, 1)
        TempSG = PreyPredatorPos(h, :);
        % S and G
        if (TempPair(1) == TempSG(1)) && (TempPair(2) == TempSG(2))
            q1 = q1 + 1;
            orderOfexp(q1, iSub) = h;
            Trace = PreyPosAtEachAttempt(:, h);
            traceMat(q1, :, iSub) = Trace;
            Trace(Trace == 0) = [];
            rt4Trace = TimingData(h, 2 : end);
            lastrt = rt4Trace(end);
            rt4Trace(end) = 0;
            sefr = find(rt4Trace == 0);
            if sefr
                rt4Trace(sefr(1)) = lastrt;
                rt4Trace(rt4Trace == 0) = [];
            end

            % save the average time the subject spends in each cell
            for fj = 1 : numel(Trace) - 1
                gridtimeMat(Trace(fj), iSub) = gridtimeMat(Trace(fj), iSub) + rt4Trace(fj);
                beenInCell(Trace(fj)) = beenInCell(Trace(fj)) + 1;
            end
        end
    end
    for bh = 1 : NumOfState
        if gridtimeMat(bh, iSub) ~= 0
            gridtimeMat(bh, iSub) = gridtimeMat(bh, iSub) ./ beenInCell(bh);
        end
    end
end


gridtimeMat(TempPair(1), :) = 0;
p = mean(gridtimeMat');
figure; bar(p)






