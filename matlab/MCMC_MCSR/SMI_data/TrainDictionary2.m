clc;
close all;
clear all;

global GInvParam;
% load GInvParam;
global GSparseInvParam;
% load GSparseInvParam;

GSparseInvParam.sizeAtom = 30;              % ��ʼ��ԭ�Ӵ�С
GSparseInvParam.nAtom = 1800;                % ԭ�Ӹ���
GSparseInvParam.trainCutInterval = 1;       % ѵ��ʱ�����и���
GSparseInvParam.trainSparsity = 20;
GSparseInvParam.trainIterNum = 20;         % ѵ����������
GSparseInvParam.isShowRebuildResult = 0;
GSparseInvParam.trainFiltCoef = 1;        % ѵ�������˲��̶�
GSparseInvParam.isShowIterInfo = 0;


%% ����׼��
load('.\SeismicData\seismic_data.mat');   %seismic_data
load('.\Load_WellData\Xline_Inline_number.mat');
Xline=Xline_Inline_number(1,:);
Inline=Xline_Inline_number(2,:);
seismic_temp=[];
for i=1:52
    seismic_temp=[seismic_temp,seismic_data(:,110*(Inline(1,2*i-1)-1)+Xline(1,2*i-1))]; 
end

load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%���ز⾮ֵ
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%���ز⾮ʱ�����
well_temp=zeros(2001,52);

for K1=1:52;
    for i=1:503
        if Well_timeData(i,2*K1-1)==fix(Well_timeData(i,2*K1-1))&&Well_timeData(i,2*K1-1)~=0
            well_temp(fix(Well_timeData(i,2*K1-1))+1,K1)=Well_valueData(i,2*K1-1);
        end
    end
end

well=zeros(70,52);
for j=1:52
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
    well(:,j)=well_temp(initial_point:initial_point+69,j);
end


[sampNum, traceNum] = size(well);

sizeAtoms = [30];       %%ԭ�Ӵ�С
nAtoms = [1800];        %%ԭ�Ӹ���
if ~exist('./Dictionaries', 'dir')
    mkdir('./Dictionaries');
end

nTrainWell = traceNum;


 for jSizeAtom = 1 : length(sizeAtoms);   %%�Բ�ͬ��ԭ�Ӵ�Сѭ��
    GSparseInvParam.sizeAtom = sizeAtoms(jSizeAtom);

    for kNAtom = 1 : length(nAtoms);       %%�Բ�ͬ��ԭ�Ӹ���ѭ��

        GSparseInvParam.nAtom = nAtoms(kNAtom); %%ԭ�Ӵ�С
        %%���������ֵ���ļ���
        dicSavePath = sprintf('./Dictionaries/DIC_%d_%d_NWell_%d_filt_%.2f.mat', GSparseInvParam.sizeAtom, GSparseInvParam.nAtom, nTrainWell, GSparseInvParam.trainFiltCoef);

        fprintf('jSizeAtom=%d, kNAtom=%d %s\n\n', jSizeAtom, kNAtom, dicSavePath);
        %%%ѵ���ֵ�
        [DIC] = stpSparseDictionaryLearn1D_2(well, GSparseInvParam.sizeAtom, GSparseInvParam.nAtom, ...
            GSparseInvParam.trainCutInterval, GSparseInvParam.trainIterNum, ...
            GSparseInvParam.trainFiltCoef, GSparseInvParam.trainSparsity, GSparseInvParam.isShowRebuildResult);
        %%%�����ֵ�
        save(dicSavePath, 'DIC');
        stpShowDictionary(DIC);
    end

end

