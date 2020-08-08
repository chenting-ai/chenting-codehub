%% 绘制单道反演结果
%% 加载数据
clear
clc
load('.\Load_WellData\Xline_Inline_number.mat')
T0=20;
markovlen=5;
well_number=52;
K1=94;
trace=Xline_Inline_number(2,K1);
%% load
dicSavePath1 = sprintf('./20200320result/MTDL2_1_50_2000/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52.mat', markovlen, T0);
saveBestX1 = load(dicSavePath1);      %%加载结果
saveBestX1=saveBestX1.saveBestX(:,trace);

dicSavePath2 = sprintf('./20200320result/MCMC/markovlen%d_T0%fsaveBestX1.mat', markovlen, T0);
saveBestX2 = load(dicSavePath2);      %%加载结果
saveBestX2=saveBestX2.saveBestX(:,trace);

% dicSavePath3 = sprintf('./ResultsData52_horizon_move/MTDL2_1_25_2340/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52horizon25_2340.mat', markovlen, T0);
% saveBestX3 = load(dicSavePath3);      %%加载结果
% saveBestX3=saveBestX3.saveBestX(:,trace);

%%
%%第99个inline 68Xline   是第39口井
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
well=zeros(2001,1);

for i=1:503
    if Well_timeData(i,K1)==fix(Well_timeData(i,K1))&&Well_timeData(i,K1)~=0
        well(fix(Well_timeData(i,K1))+1,1)=Well_valueData(i,K1);
    end
end
% well(818,1)=0;
% well(1035,1)=0;
%% 检测有效数据长度
for sampling_number=1:2001      
    if( saveBestX1(sampling_number,1)~=0)
        initial_point=sampling_number;
        break;
    end
end
for sampling_number=sampling_number:2001   
    if( saveBestX1(sampling_number,1 )~=0)
       termination_point = sampling_number;
    end
end
useful_len = termination_point-initial_point+1;
%纵向滤波作为初始模型
trueModel = well(initial_point:termination_point,1);
saveBestX1 = saveBestX1(initial_point:termination_point,1);
saveBestX2 = saveBestX2(initial_point:termination_point,1);
% saveBestX3 = saveBestX3(initial_point:termination_point,1);
%%
options = {'(a)','(b)','(c)','(d)'};
t =1:useful_len;
linewidth = 1.5;
%% 
figure('color',[1 1 1]);set(gcf, 'position', [200 200 300 700]);
plot(trueModel/1000, t , 'g', 'linewidth', linewidth);hold on;
plot(saveBestX1/1000, t , 'r--', 'linewidth', linewidth);hold on;
plot(saveBestX2/1000, t , 'k', 'linewidth', linewidth);hold on;
% plot(saveBestX3/1000, t , 'k', 'linewidth', linewidth);hold on;
% plot(horizon/1000, t , 'r', 'linewidth', linewidth);hold on;
set(gca, 'xlim', [4.5 9.5]);
set(gca,'YTick', 1:10:useful_len);
set(gca,'YTickLabel', 905:10:964);
xlabel('Impedance(10^3*g/cc*m/s)');
set(gca, 'ydir','reverse');
ylabel('Time(ms)');
% legend({'Well', 'MCMC', 'MCDL'});
title('W3-2104');
a1=corr(trueModel,saveBestX1);
a2=sum((trueModel-saveBestX1).*(trueModel-saveBestX1))/(useful_len);
b1=corr(trueModel,saveBestX2);
b2=sum((trueModel-saveBestX2).*(trueModel-saveBestX2))/(useful_len);
name= sprintf('T0_%dmarkovlen%d反演结果井%dMCDLcor%f_MSE%f_MCMCcor%f_MSE%f',T0,markovlen,K1,a1,a2,b1,b2);
% path = sprintf('F:/matlab临时图片存储/0323论文用图/MCDL_2_50_2000'); 
% fileName = sprintf('%s/%s.tif', path, name);
% print('-dtiff', '-r600', fileName);