clc;
close all;
clear all;

global GInvParam;
% load GInvParam;
global GSparseInvParam;
% load GSparseInvParam;

GSparseInvParam.sizeAtom = 30;              % 初始化原子大小
GSparseInvParam.nAtom = 2000;                % 原子个数
GSparseInvParam.trainCutInterval = 1;       % 训练时样本切割间隔
GSparseInvParam.trainSparsity = 10;
GSparseInvParam.trainIterNum = 20;         % 训练迭代次数
GSparseInvParam.isShowRebuildResult = 0;
GSparseInvParam.trainFiltCoef = 1;        % 训练样本滤波程度
GSparseInvParam.isShowIterInfo = 0;

load('.\SeismicData\seismic_data.mat');   %seismic_data
load('.\Load_WellData\Xline_Inline_number.mat');
%%
Xline=Xline_Inline_number(1,:);
Inline=Xline_Inline_number(2,:);
seismic_temp=[];
for i=1:104
    seismic_temp=[seismic_temp,seismic_data(:,110*(Inline(1,i)-1)+Xline(1,i))]; 
end

load('E:\matlab_project\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('E:\matlab_project\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
well_temp=zeros(2001,104);

for K1=1:104;
    for i=1:503
        if Well_timeData(i,K1)==fix(Well_timeData(i,K1))&&Well_timeData(i,K1)~=0
            well_temp(fix(Well_timeData(i,K1))+1,K1)=Well_valueData(i,K1);
        end
    end
end

well=zeros(62,104);
for j=1:104
    for sampling_number=1:2001      
        if( seismic_temp(sampling_number,j)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:2001  
        if( seismic_temp(sampling_number,j )~=0)
           termination_point=sampling_number;
        end
    end
    useful_len = termination_point-initial_point+1;
    well(1:useful_len,j)=well_temp(initial_point:termination_point,j);
end



[sampNum, traceNum] = size(well);

sizeAtoms = [30];       %%原子大小
nAtoms = [1500];        %%原子个数
if ~exist('./Dictionaries', 'dir')
    mkdir('./Dictionaries');
end

% nTrainWell = traceNum;
% trainLogs = (Well_valueData1);


nTrainWell = traceNum;
 for jSizeAtom = 1 : length(sizeAtoms);   %%对不同的原子大小循环
    GSparseInvParam.sizeAtom = sizeAtoms(jSizeAtom);

    for kNAtom = 1 : length(nAtoms);       %%对不同的原子个数循环

        GSparseInvParam.nAtom = nAtoms(kNAtom); %%原子大小
        %%命名储存字典的文件名
        dicSavePath = sprintf('./Dictionaries/DIC_%d_%d_NWell_%d_filt_%.2f.mat', GSparseInvParam.sizeAtom, GSparseInvParam.nAtom, nTrainWell, GSparseInvParam.trainFiltCoef);

        fprintf('jSizeAtom=%d, kNAtom=%d %s\n\n', jSizeAtom, kNAtom, dicSavePath);
        %%%训练字典
        [DIC] = stpSparseDictionaryLearn1D(well, GSparseInvParam.sizeAtom, GSparseInvParam.nAtom, ...
            GSparseInvParam.trainCutInterval, GSparseInvParam.trainIterNum, ...
            GSparseInvParam.trainFiltCoef, GSparseInvParam.trainSparsity, GSparseInvParam.isShowRebuildResult);
        %%%保存字典
        save(dicSavePath, 'DIC');
        stpShowDictionary(DIC);
    end

end

