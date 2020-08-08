%% 绘制单道反演结果
%% 加载数据
clear
clc
load('.\Load_WellData\Xline_Inline_number.mat')
T0=20;
markovlen=5;


well_number=52;
K1=79;
trace=Xline_Inline_number(1,K1);
impedence=load('.\SeismicData\inline99_initalmodel_52.mat'); 
horizon=impedence.initial_impedence(:,trace);  

% dicSavePath2 = sprintf('./ResultsData_simple_richer/add_syn/MTDL_2well_39_trace_68/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52.mat', markovlen, T0);
% nohorizon = load(dicSavePath2);      %%加载结果
% nohorizon=nohorizon.saveBestX;

dicSavePath3 = sprintf('./ResultsData52_nohorizon_move/MCMC/markovlen%d_T0%fsaveBestX1.mat', markovlen, T0);
MCMC = load(dicSavePath3);      %%加载结果
MCMC=MCMC.saveBestX;


%%第99个inline 68Xline   是第39口井
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
well=zeros(2001,1);

for i=1:503
    if Well_timeData(i,K1)==fix(Well_timeData(i,K1))&&Well_timeData(i,K1)~=0
        well(fix(Well_timeData(i,K1)),1)=Well_valueData(i,K1);
    end
end
% well(818,1)=0;
% well(1035,1)=0;


%% 检测有效数据长度
for sampling_number=1:2001      
    if( horizon(sampling_number,1)~=0)
        initial_point=sampling_number;
        break;
    end
end
for sampling_number=sampling_number:2001   
    if( horizon(sampling_number,1 )~=0)
       termination_point = sampling_number;
    end
end
useful_len = termination_point-initial_point+1;
%纵向滤波作为初始模型
trueModel = well(initial_point:termination_point,1);
horizon = horizon(initial_point:termination_point,1);
% nohorizon = nohorizon(initial_point:termination_point,1);
MCMC = MCMC(initial_point:termination_point,1);
%%
options = {'(a)','(b)','(c)','(d)'};
t =1:useful_len;
linewidth = 2.5;
%% 
figure('color',[1 1 1]);set(gcf, 'position', [200 200 300 700]);
plot(trueModel/1000, t , 'k--', 'linewidth', linewidth);hold on;
plot(horizon/1000, t , 'r', 'linewidth', linewidth);hold on;
% plot(MCMC/1000, t , 'r', 'linewidth', linewidth);hold on;
set(gca, 'xlim', [4 9]);
set(gca,'YTick', 1:10:useful_len);
set(gca,'YTickLabel', initial_point:10:termination_point);
xlabel('Impedance(10^3*g/cc*m/s)');
set(gca, 'ydir','reverse');
ylabel('Time(ms)');
% title('判别字典');
legend({'well', 'simulation'});
% a1=corr(trueModel,horizon);
% a2=sum((trueModel-horizon).*(trueModel-horizon))/(useful_len);
% name= sprintf('T0_%dmarkovlen%dMCDLhorizon52反演结果+只用到了合成地震记录cor%f_MSE%f',T0,markovlen,a1,a2);
% path = sprintf('F:/matlab临时图片存储/第四轮/%d单井对比第39井第68道-ricker/MCDL_2horizon',well_number); 
% % stpSaveFigure(path, name);


% 
% figure;set(gcf, 'position', [500 200 300 700]);
% plot(trueModel/1000, t , 'k--', 'linewidth', linewidth);hold on;
% plot(nohorizon/1000, t , 'g', 'linewidth', linewidth);hold on;
% set(gca, 'xlim', [4 9]);
% set(gca,'YTick', 1:10:useful_len);
% set(gca,'YTickLabel', initial_point:10:termination_point);
% xlabel('Impedance(10^3*g/cc*m/s)');
% set(gca, 'ydir','reverse');
% ylabel('Time(ms)');
% title('普通字典');
% legend({'well','nohorizon'});
% b1=corr(trueModel,nohorizon);
% b2=sum((trueModel-nohorizon).*(trueModel-nohorizon))/(useful_len);
% name= sprintf('T0_%dmarkovlen%dMCDLhorizon52反演结果+只用到了合成地震记录cor%f_MSE%f',T0,markovlen,b1,b2);
% path = sprintf('F:/matlab临时图片存储/第四轮/%d单井对比第39井第68道-ricker/MCDL_2',well_number); 
% % stpSaveFigure(path, name);
% 
% 
% 
% figure;set(gcf, 'position', [800 200 300 700]);
% plot(trueModel/1000, t , 'k--', 'linewidth', linewidth);hold on;
% plot(MCMC/1000, t , 'b', 'linewidth', linewidth);hold on;
% legend({'well', 'MCMC'});
% set(gca, 'xlim', [4 9]);
% xlabel('Impedance(10^3*g/cc*m/s)');
% set(gca,'YTick', 1:10:useful_len);
% set(gca,'YTickLabel', initial_point:10:termination_point);
% set(gca, 'ydir','reverse');
% ylabel('Time(ms)');
% title('MCMC');
% c1=corr(trueModel,MCMC);
% c2=sum((trueModel-MCMC).*(trueModel-MCMC))/(useful_len);
% name= sprintf('T0_%dmarkovlen%dMCDLhorizon52反演结果+只用到了合成地震记录cor%f_MSE%f',T0,markovlen,c1,c2);
% path = sprintf('F:/matlab临时图片存储/第四轮/%d单井对比第39井第68道-ricker/MCMC',well_number); 
% % stpSaveFigure(path, name);
% 
% 
% 
% 
% 
% 
% 
% fprintf('a1=%f,b1=%f,c1=%f\n',a1,b1,c1 );
% fprintf('a2=%f,b2=%f,c2=%f',a2,b2,c2 );
% 
