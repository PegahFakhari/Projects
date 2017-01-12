% Freq plot with colormap
clc
clear
close all
% ======== design the Environment ======== %
NumOfBlocks = 10; % for experiment 3 we have 10 blocks.
TestPhaseNum = 8;
PrestestBlock = 7;
NumOfTrials = 20;
MaxNumPerEps = 15;
GridSize = [4, 7];
NumOfState = prod(GridSize);
NumOfAction = 4;
FixedLock = [1, 2, 4, 10, 18, 24, 25, 26, 28]; % !!#########!!
% FixedLock = [1, 2, 4, 10, 14, 18, 24, 25, 26, 28]; For Exp 1
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
NumOfSubj  = 32; % 19 for Exp 1, 36 for Exp2, 32 for Exp 3
% ================================= %
% ========= MLE parameter =========== %
Discount = 0.95;
Temperature = 9;
Matrix1 = zeros(3*4, 3*7);

for iSub = 1 : NumOfSubj
    clc
    fprintf('sim=%d', iSub, NumOfSubj);
    ID = num2str(iSub);
    %     SubjectID = strcat('mdfs', ID);
    %     SubjectID = strcat('Exp1S', ID);
    SubjectID = strcat('STEN', ID, '-Whole'); %mdfs means I added the final position to the preypredatorposition

    load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Data\Paid Subjects With 10 Blocks\', SubjectID,'.mat'))
    PreyPosInTestPhase = ...
        PreyPosAtEachAttempt(:, (NumOfBlocks - 3) * NumOfTrials + 1: NumOfBlocks * NumOfTrials);
    TestPairs = PreyPredatorPos((NumOfBlocks - 3) * NumOfTrials + 1: NumOfBlocks * NumOfTrials, :);
    TestActionUp = UpChoice(PrestestBlock + 1 : NumOfBlocks, :, :);
    TestActionRight = RightChoice(PrestestBlock + 1 : NumOfBlocks, :, :);
    TestActionDown = DownChoice(PrestestBlock + 1 : NumOfBlocks, :, :);
    TestActionLeft = LeftChoice(PrestestBlock + 1 : NumOfBlocks, :, :);
    % ======== Test Block ======== %
    for jt = 1 : size(PreyPosInTestPhase, 2)
        if ObsKindVecIn3Block(jt) == 1
            % for experiment 3 only:
            % choose one pair of your interest
            if TestPairs(jt, 1) == 3 && TestPairs(jt, 2) == 27
                currpath = PreyPosInTestPhase(:, jt);
                currpath(currpath == 0) = [];
                if numel (currpath) == MaxNumPerEps + 1;
                    currpath(end) = [];
                end
                for m = 1 : numel (currpath)
                    [i,j] = ind2sub(GridSize,currpath(m));
                    Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 2) = Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 2) + 1;
                    if jt <= NumOfTrials
                        Matrix1( (i - 1)  * 3 + 1, (j - 1) * 3 + 2) = ...
                            Matrix1( (i - 1)  * 3 + 1, (j - 1) * 3 + 2) + TestActionUp(1, jt, m) ; % Going up
                        Matrix1( (i - 1)  * 3 + 3, (j - 1) * 3 + 2) = ...
                            Matrix1( (i - 1)  * 3 + 3, (j - 1) * 3 + 2) + TestActionDown(1, jt, m); % Going Down
                        Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 1) = ...
                            Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 1) + TestActionLeft(1, jt, m); % Going Left
                        Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 3) = ...
                            Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 3) + TestActionRight(1, jt, m); % Going Right
                    else
                        if jt > NumOfTrials && jt <= 2*NumOfTrials
                            jtm = jt - NumOfTrials;
                            Matrix1( (i - 1)  * 3 + 1, (j - 1) * 3 + 2) = ...
                                Matrix1( (i - 1)  * 3 + 1, (j - 1) * 3 + 2) + TestActionUp(2, jtm, m) ; % Going up
                            Matrix1( (i - 1)  * 3 + 3, (j - 1) * 3 + 2) = ...
                                Matrix1( (i - 1)  * 3 + 3, (j - 1) * 3 + 2) + TestActionDown(2, jtm, m); % Going Down
                            Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 1) = ...
                                Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 1) + TestActionLeft(2, jtm, m); % Going Left
                            Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 3) = ...
                                Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 3) + TestActionRight(2, jtm, m); % Going Right
                        else
                            jtm = jt - 2*NumOfTrials;
                            Matrix1( (i - 1)  * 3 + 1, (j - 1) * 3 + 2) = ...
                                Matrix1( (i - 1)  * 3 + 1, (j - 1) * 3 + 2) + TestActionUp(3, jtm, m) ; % Going up
                            Matrix1( (i - 1)  * 3 + 3, (j - 1) * 3 + 2) = ...
                                Matrix1( (i - 1)  * 3 + 3, (j - 1) * 3 + 2) + TestActionDown(3, jtm, m); % Going Down
                            Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 1) = ...
                                Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 1) + TestActionLeft(3, jtm, m); % Going Left
                            Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 3) = ...
                                Matrix1( (i - 1)  * 3 + 2, (j - 1) * 3 + 3) + TestActionRight(3, jtm, m); % Going Right
                        end
                    end
                end
            end
            if Matrix1(8, 6) ~= 0
                clc
            end
        end
    end
end

Matrix1 = 1 .* Matrix1;
Matrix1(Matrix1==0)=NaN;
a = [Matrix1 NaN(12,1)];
a = [a; NaN(1,22)];

figure
h = pcolor(a');
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])

% Add Vertical Lines
for i = 1 : 3 : 22
    hold on
    line( [13, 1], [i,i], 'LineWidth',4, 'Color',[0 0 0])
end

% Add Horizontal Lines
for i = 1 : 3 : 22
    hold on
    line( [i,i], [22, 1], 'LineWidth',4, 'Color',[0 0 0])
end

set(h, 'EdgeColor', 'none');


