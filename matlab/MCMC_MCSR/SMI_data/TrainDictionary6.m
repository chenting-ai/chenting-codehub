clc;
close all;
clear all;


%% 数据准备
load('.\SeismicData\seismic_data.mat');   %seismic_data
load('.\Load_WellData\Xline_Inline_number.mat');
Xline=Xline_Inline_number(1,:);
Inline=Xline_Inline_number(2,:);
seismic_temp=[];
for i=1:52
    seismic_temp=[seismic_temp,seismic_data(:,110*(Inline(1,2*i-1)-1)+Xline(1,2*i-1))]; 
end

load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
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
%% 数据标签处理
sizeAtoms = 20;       %%原子大小
classifier_number = 9;    %类别数量
training_feats = [];
H_train = [];
k=1;
% figure('color',[1 1 1])
% imagesc(well);
% name= sprintf('井数据');
% path = sprintf('F:/matlab临时图片存储/第三轮/目的层位测井数据52口井'); 
% colorbar
% % stpSaveFigure(path, name);


for i = 1 :1: size(well,1)
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

training_feats=training_feats(:,625:2964);
H_train=H_train(:,625:2964);
%% 训练字典
dictsize = 450;        %%原子个数
sparsitythres = 20; % sparsity prior
sqrt_alpha = 15; % weights for label constraint term
sqrt_beta = 10; % weights for classification err term
iterations = 5; % iteration number
iterations4ini = 50; % iteration number for initialization

dicSavePath = sprintf('./well_plot/sparsitythres20_Dictionary/dictionarydataWell_Num52_1M_%dL_%d.mat', sizeAtoms,dictsize);
load(dicSavePath);
dicSavePath1 = sprintf('./well_plot/sparsitythres20_Dictionary/dictionarydataWell_Num52_2M_%dL_%d.mat', sizeAtoms,dictsize);
load(dicSavePath1);





%% classification process
[MSE] = test_MSEF(D1, training_feats, sparsitythres);
fprintf('\nFinal MSE for LC-KSVD1 is : %.03f ', MSE);

[MSE2] = test_MSEF(D2, training_feats, sparsitythres);
fprintf('\nFinal MSE for LC-KSVD2 is : %.03f ', MSE2);





















% for i = sizeAtoms/2 : size(well,1)-sizeAtoms/2
%     if i>size(well,1)-sizeAtoms/2
%        jj=size(well,1)-sizeAtoms/2;
%     elseif i<sizeAtoms/2
%        jj=sizeAtoms/2;
%     else
%        jj=i;
%     end
%     temp_H=zeros(classifier_number,1);
%     temp_H(jj-sizeAtoms/2+1,1)=1;
%     for j = 1 : size(well,2)
%         training_feats = [training_feats, well(jj-sizeAtoms/2+1:jj+sizeAtoms/2,j)];
%         H_train = [H_train,temp_H];
%     end
% end
