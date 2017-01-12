% in This file I read each subject's data and then calculate the mean
% reward in each block from 1 to 7. So for each subject I have 7 data
% reward point

clc
clear
close all
NumOfTrials = 20;
NumBlcks = 7;
NumOfSub = 32;
AveRewpSubpBlock = zeros(NumOfSub * NumBlcks, 3);
for z = 1 : NumOfSub
    clc
    fprintf('sim=%d', z);    
    ID = num2str(z);
    SubjectID = strcat('STEN', ID,'-Whole');
    load(strcat('C:\Users\pegah\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Data\Paid Subjects With 10 Blocks\', SubjectID,'.mat'))
    for kh = 1 : NumBlcks
        AveRewpSubpBlock( (z-1) * NumBlcks + kh, 1) = z;
        AveRewpSubpBlock( (z-1) * NumBlcks + kh, 2) = sum(sum((ImRewardData(:, (kh-1) * NumOfTrials + 1: (kh) * NumOfTrials))))/NumOfTrials;
        AveRewpSubpBlock( (z-1) * NumBlcks + kh, 3) = kh;
    end
end

%===determine the directory===
a= 'FitWithMLE\PrepareAveR4LongitudinalAnalysis.m';
Directory = which(a);
Directory = Directory(1:(end-numel(a)));
%===determine the directory===
Result = 'RewardData';

header = {'Subject','RR', 'blockNo'};
data = num2cell(AveRewpSubpBlock);

xlswrite(strcat(Directory, 'FitWithMLE\', Result,'.xls'),[header;data]);





















