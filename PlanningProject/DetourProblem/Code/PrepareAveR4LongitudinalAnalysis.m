% in This file I read each subject's data and then calculate the mean
% reward in each block from 1 to 7. So for each subject I have 7 data
% reward point

clc
clear
close all
NumOfTrials = 20;
NumBlcks = 7;
NumOfSub = 34;
Ind2Delete = [25, 21, 15, 12, 6];
AveRewpSubpBlock = zeros(NumOfSub * NumBlcks, 3);
for z = 1 : NumOfSub
    load(strcat('C:\Users\pfakhari\Dropbox\Busemeyer Lab\NSF Project\Reinforcement Learning\Model Fitting Codes\Replanning In Decision Tree\Data\Paid Subjects\S', num2str(z)))
    for kh = 1 : NumBlcks
        AveRewpSubpBlock( (z-1) * NumBlcks + kh, 1) = z;
        AveRewpSubpBlock( (z-1) * NumBlcks + kh, 2) = sum(sum((ImRewardData(:, (kh-1) * NumOfTrials + 1: (kh) * NumOfTrials))))/NumOfTrials;
        AveRewpSubpBlock( (z-1) * NumBlcks + kh, 3) = kh;
    end
end

for df = Ind2Delete
    AveRewpSubpBlock(((df - 1)* NumBlcks + 1: df* NumBlcks), :, :) =[];
end

%===determine the directory===
a= 'ModelBasedFitWithMLE\PrepareAveR4LongitudinalAnalysis.m';
Directory = which(a);
Directory = Directory(1:(end-numel(a)));
%===determine the directory===
Result = 'Data4LongAnalysis';

header = {'Subject','RR', 'blockNo'};
data = num2cell(AveRewpSubpBlock);

xlswrite(strcat(Directory, 'ModelBasedFitWithMLE\', Result,'.xls'),[header;data]);





















