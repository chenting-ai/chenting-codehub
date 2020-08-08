%% չʾ�����ߣ��Լ����ֵ��е�ԭ������
clc;
clear all;

%% ���ؾ�����
public_path='E:\matlab_project\The_stochastic_inversion\NewDictionary_MCMC_inversion_move4\marmousi_SM\';
well_path=[public_path,'usefuldata\marmousi2_SM.mat'];%marmousi2_SM
load(well_path);
well=marmousi2_SM(:,1:25:end);


well_idx=1:2:14; %% ��7�ھ�������
idx = [4,1,7,5,2,6,3];
well_plot=log(well(:,well_idx(idx)));

%% ��������
figure('color',[1 1 1]);
set(gcf, 'position', [100 100 600 500]);
linewidth=0.5;
for i=1:size(well_plot,2)
    
    one_well=well_plot(:,i);
    well_max=max(one_well);
    well_min=min(one_well);
    one_well=(one_well-well_min)/(well_max-well_min)+0.6*i;
    t=1:size(one_well);
    plot(one_well,t,'k', 'linewidth', linewidth);
    set(gca, 'ydir','reverse');
    set(gca, 'ylim', [1 200]);
    set(gca, 'xlim', [0.5 5.3]);
%     set(gca,'xTickLabel', 1:1:5);
    ylabel('Sample number');
    set(gca,'xtick',[],'xticklabel',[])
    hold on;
end

%% ���ֵ��е�ԭ��
T=20;
well_path=[public_path,'Dictionaries\marmousi14DIC_30_1500_NWell_14_filt_1.00.mat'];%marmousi2_SM
load(well_path);   %%DIC
Dic_plot=[];
%��һ�ھ�������85:114��ԭ����6
temp1=well_plot(:,1);
temp_spare1=omp(DIC,temp1(85:114),DIC'*DIC,T);
[~,Dic_idx]=max(temp_spare1);
Dic_plot=[Dic_plot,DIC(:,Dic_idx-5:Dic_idx+4)];
%�ڶ��ھ�������20:49��ԭ����2
temp1=well_plot(:,2);
temp_spare1=omp(DIC,temp1(20:49),DIC'*DIC,T);
[~,Dic_idx]=max(temp_spare1);
Dic_plot=[Dic_plot,DIC(:,Dic_idx-1:Dic_idx+8)];
%
Dic_plot=[Dic_plot,DIC(:,Dic_idx+120:Dic_idx+129)];
%���Ŀھ�������171:200��ԭ����9
temp1=well_plot(:,4);
temp_spare1=omp(DIC,temp1(171:200),DIC'*DIC,T);
[~,Dic_idx]=max(temp_spare1);
Dic_plot=[Dic_plot,DIC(:,Dic_idx-8:Dic_idx+1)];
%����ھ�������41:70��ԭ����5
temp1=well_plot(:,5);
temp_spare1=omp(DIC,temp1(41:70),DIC'*DIC,T);
[~,Dic_idx]=max(temp_spare1);
Dic_plot=[Dic_plot,DIC(:,Dic_idx-4:Dic_idx+5)];
%�����ھ�������116:145��ԭ����1
temp1=well_plot(:,6);
temp_spare1=omp(DIC,temp1(116:145),DIC'*DIC,T);
[~,Dic_idx]=max(temp_spare1);
Dic_plot=[Dic_plot,DIC(:,Dic_idx:Dic_idx+9)];

stpShowDictionary(Dic_plot);





