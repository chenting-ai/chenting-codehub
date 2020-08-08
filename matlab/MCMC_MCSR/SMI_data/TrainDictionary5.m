clc;
close all;
clear all;
%% ѵ���ֵ�


global GInvParam;
% load GInvParam;
global GSparseInvParam;
% load GSparseInvParam;

GSparseInvParam.sizeAtom = 26;              % ��ʼ��ԭ�Ӵ�С
GSparseInvParam.nAtom = 200;                % ԭ�Ӹ���
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

load('E:\matlab_project\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%���ز⾮ֵ
load('E:\matlab_project\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%���ز⾮ʱ�����
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
%% ���ݱ�ǩ����
sizeAtoms = 26;       %%ԭ�Ӵ�С
classifier_number = 9;    %�������
training_feats = [];
H_train = [];
k=1;
% figure('color',[1 1 1])
% imagesc(well);
% name= sprintf('������');
% path = sprintf('F:/matlab��ʱͼƬ�洢/������/Ŀ�Ĳ�λ�⾮����52�ھ�'); 
% colorbar
% % stpSaveFigure(path, name);


for i = sizeAtoms/2 :1: size(well,1)-sizeAtoms/2
    if i>size(well,1)-sizeAtoms/2
       jj=size(well,1)-sizeAtoms/2;
    elseif i<sizeAtoms/2
       jj=sizeAtoms/2;
    else
       jj=i;
    end
    temp_Hor=[];
    temp_Hor=[temp_Hor;abs(i-15);abs(i-20);abs(i-25);abs(i-30);abs(i-35);abs(i-40);abs(i-45);abs(i-50);abs(i-55)];
    [~,Hor] = min(temp_Hor); 
    
    temp_H=zeros(classifier_number,1);
    temp_H(Hor,1)=1;
    for j = 1 : size(well,2)
        training_feats = [training_feats, well(jj-sizeAtoms/2+1:jj+sizeAtoms/2,j)];
        H_train = [H_train,temp_H];
    end
    
end



%%
DIC=[];
for i=1:9
    data=training_feats(:,(i-1)*260+1:i*260);
    [dic] = stpSparseDictionaryLearn1D(data, GSparseInvParam.sizeAtom, GSparseInvParam.nAtom, ...
            GSparseInvParam.trainCutInterval, GSparseInvParam.trainIterNum, ...
            GSparseInvParam.trainFiltCoef, GSparseInvParam.trainSparsity, GSparseInvParam.isShowRebuildResult);
    DIC=[DIC,dic];
end


%% test
count=0;
for j=1:260
    temp1=training_feats(:,(3-1)*260+j);
    temp_spare1=omp(DIC,temp1,DIC'*DIC,GSparseInvParam.trainSparsity);
    [a,b]=max(temp_spare1);
    if b<601&&b>400
        count=count+1;
    end
    temp1=DIC*temp_spare1;
end
corect=count/260






