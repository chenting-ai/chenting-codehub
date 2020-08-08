clc;
close all;
clear all;

global GInvParam;
% load GInvParam;
global GSparseInvParam;
% load GSparseInvParam;

GSparseInvParam.sizeAtom = 30;              % ��ʼ��ԭ�Ӵ�С
GSparseInvParam.nAtom = 1500;                % ԭ�Ӹ���
GSparseInvParam.trainCutInterval = 1;       % ѵ��ʱ�����и���
GSparseInvParam.trainSparsity = 20;
GSparseInvParam.trainIterNum = 20;         % ѵ����������
GSparseInvParam.isShowRebuildResult = 0;
GSparseInvParam.trainFiltCoef = 0.7;        % ѵ�������˲��̶�
GSparseInvParam.isShowIterInfo = 0;

public_path='E:\matlab_project\The_stochastic_inversion\NewDictionary_MCMC_inversion_move4\marmousi_SM\';
well_path=[public_path,'usefuldata\marmousi2_SM.mat'];%marmousi2_SM
load(well_path);
well=marmousi2_SM(:,1:25:end);
[sampNum, traceNum] = size(well);
sizeAtoms = [30];       %%ԭ�Ӵ�С
nAtoms = [1500];        %%ԭ�Ӹ���
if ~exist('./Dictionaries', 'dir')
    mkdir('./Dictionaries');
end

% nTrainWell = traceNum;
% trainLogs = (Well_valueData1);


nTrainWell = traceNum;
wellIndex = round(linspace(1, traceNum, nTrainWell));
trainLogs = zeros(sampNum, nTrainWell);
for i = 1 : nTrainWell
    for sampling_number=1:sampNum     
        if( well(sampling_number,i)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:sampNum   
        if( well(sampling_number,i)~=0)
           termination_point=sampling_number;
        end
    end 
    trainLogs(initial_point:termination_point, i) = log(well(initial_point:termination_point, wellIndex(i)));   
end

 for jSizeAtom = 1 : length(sizeAtoms);   %%�Բ�ͬ��ԭ�Ӵ�Сѭ��
    GSparseInvParam.sizeAtom = sizeAtoms(jSizeAtom);

    for kNAtom = 1 : length(nAtoms);       %%�Բ�ͬ��ԭ�Ӹ���ѭ��

        GSparseInvParam.nAtom = nAtoms(kNAtom); %%ԭ�Ӵ�С
        %%���������ֵ���ļ���
        dicSavePath = sprintf('./Dictionaries/marmousi14DIC_%d_%d_NWell_%d_filt_%.2f.mat', GSparseInvParam.sizeAtom, GSparseInvParam.nAtom, nTrainWell, GSparseInvParam.trainFiltCoef);

        fprintf('jSizeAtom=%d, kNAtom=%d %s\n\n', jSizeAtom, kNAtom, dicSavePath);
        %%%ѵ���ֵ�
        [DIC] = stpSparseDictionaryLearn1D(trainLogs, GSparseInvParam.sizeAtom, GSparseInvParam.nAtom, ...
            GSparseInvParam.trainCutInterval, GSparseInvParam.trainIterNum, ...
            GSparseInvParam.trainFiltCoef, GSparseInvParam.trainSparsity, GSparseInvParam.isShowRebuildResult);
        %%%�����ֵ�
        save(dicSavePath, 'DIC');
        stpShowDictionary(DIC);
    end

end

