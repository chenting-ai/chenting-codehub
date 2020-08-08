% 测试一维模型反演
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
close all;

%% 制作模型数据
load('./Load_WellData/marmousi_imp_horizon_350_377.mat');

% 正演
waveletFreq = 30;
wave = s_create_wavelet({'type','ricker'}, {'frequencies',waveletFreq}, {'step', 1}, {'wlength',200});
wavelet = wave.traces;
% si = 160;
si = 1;
trueModel = trueModel(si:end, :);
sampNum = size(trueModel, 1);
[poststack, G] = csGenPost1DForwardModel(wavelet, trueModel, sampNum, sampNum-1 );
% 添加噪声
noiseType = 1;
noiseDB = 4;
noiseOption = {'gs','lp'};
postNoise = wyjGenNormNoise2D(poststack, noiseDB, noiseType);
% 取道
traceId = 11;
Model.G = G; % 正演矩阵 
Model.d0 = poststack(:, traceId);
Model.m = trueModel(:, traceId);
Model.d = postNoise(:, traceId);
% 
% modelName = ['model-' noiseOption{noiseType} '-' sprintf('%d', noiseDB) 'DB.mat'];
% save([ './data/' modelName],  'Model');
% snr = SNR_singlech(Model.d0, Model.d);
%% 加载模型

% load('./data/model-gs-6DB.mat');
[sampNum, ~] = size(Model.m);
% Model.d = Model.d0;
% 初始模型 
Wn = 0.03;
initModel = csFilterProfile(trueModel, Wn, 'v'); % 纵向滤波
initModel = csFilterProfile(initModel, Wn, 'h'); % 横向滤波
u0 = log(initModel(:, traceId));
% 添加噪声 , 也可以使用model里已有的噪声记录
% [postNoise ,Noise_lp_6]= wyjGenNormNoise(Model.d0, 6, 2); % 用王老师的函数
% [postNoise ,Noise_lp_6]= stpGenNormNoise(Model.d0, 6, 2); % 用厍斌的函数
%% 设置参数
% 输出符号信息
typeMethods = {'GD', 'BFGS', 'CG', 'NCG', 'VSSNSGA', 'csNCG'};  % 迭代方法(1-梯度下降 2-拟牛顿法 3-共轭梯度法 4-厍斌拟共轭梯度 5-符号梯度法 6-自适应拟共轭)
typeObj = {'L1', 'L2', 'huber', 'MCC','MCCi', 'SAM', 'GEV'};  % 目标函数(1-L1,2-L2,3-huber,4-我写的Mcc,5-厍斌写的Mcci,6-混合范数,7-广义极值范数)    
typeReg = {'none', 'Tic1', 'TV', 'MGS'};
testItems = {
 % 迭代算法，目标函数，正则项，正则化参数lamda,迭代次数，MGS的sigma值，MCC的theta值

% 6db 高斯噪声 无正则化
%     [3 2 1 0  30     0.0001 0.001];

% 6db 高斯噪声 tikhonov正则化
%     [3 2 2 0.05  30     0.0001 0.001];
%     [3 2 2 0.5  30     0.0001 0.001];
%     [3 2 2 5 80     0.0001 0.001];
% 6db 高斯噪声 正则化对比

     [4 2 2 5  30     0.0001 0.001];  %Tikhonov
     [4 2 3 0.05  150     0.0001 0.001]; %TV

%      [4 2 4 0.005  30     0.04  0.001];   %MGS
%      [4 2 4 0.005  30     0.001  0.001];
%      [4 2 4 0.005  30     0.0003  0.001];
%       [4 2 4 0.003  30     0.0003  0.001];
    };
dt = 2;
nItems = length(testItems);
color = {'r', 'k', 'b--', 'g', 'm--', 'y', 'b', 'r--', 'g--',  'k--'};
tstr = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)'};

%% 绘制真实模型
if 0
linewidth = 1.5;
figure(1);

set(gcf, 'position', [100 100 430 520]);
% 1
t1 = 1 : 1: length(Model.m);
subplot(1, 2 ,1);plot(Model.m/1000, t1, 'b', 'linewidth', linewidth);
set(gca, 'xlim', [min(Model.m/1000)-0.5 max(Model.m/1000)+0.5]);

set(gca, 'ylim', [0 t1(end)]);
set(gca, 'ydir','reverse');
set(gca,'YTick', 20:40:length(Model.m));
set(gca,'YTickLabel',1000:500:3000);

% text(7.4, 205, '波阻抗','fontsize',10);
% text(6.0, 212, '(10^3*g/cc*m/s)','fontsize',10);
% text(1.8, 116, 'Time(ms)','fontsize',12,  'rotation', 90);

text(7.2, 375, '波阻抗','fontsize',10);
text(5.8, 388, '(10^3*g/cc*m/s)','fontsize',10);
text(0.2, 216, 'Time(ms)','fontsize',12,  'rotation', 90);

% title('(a)','fontsize',14, 'position',[9,-3]);
title('(a)','fontsize',14, 'position',[8.2,-3]);
% 2
t2 = 1 : 1: length(Model.d);
subplot(1, 2 ,2);
plot(Model.d0, t2, 'b', 'linewidth',linewidth);hold on;
plot(Model.d, t2, 'r--', 'linewidth', linewidth);
legend('无噪声','含噪声');

set(gca, 'xlim', [-0.6 0.6]);

set(gca, 'ylim', [0 t2(end)]);
set(gca, 'ylim', [0 t1(end)]);
set(gca, 'ydir','reverse');
set(gca,'YTick', 20:40:length(Model.d));
set(gca,'YTickLabel',1000:500:3000);

% text(-0.2, 205, '振幅','fontsize',10);
text(-0.1, 375, '振幅','fontsize',10);

title('(b)','fontsize',14, 'position',[-0.0,-3]);
end
%% 开始反演
strs = cell(nItems+1, 1);
result = zeros(sampNum, nItems);
res = cell(1, nItems);
for iItem = 1 : nItems
    item = testItems{iItem};
    strs{iItem} = sprintf('%s-%s-%s-lamda:%.3f-sigma:%.4f', typeMethods{item(1)}, ...
        typeObj{item(2)}, typeReg{item(3)}, item(4), item(6));
    % 求解
%     [result(:,iItem), res{iItem}] = csLinearSolver(Model.d, Model.d0, Model.G, u0, log(Model.m), item);       
    [result(:,iItem), res{iItem}] = csLinearSolver(Model.d, [], Model.G, u0, [], item);       
                                             %    噪声记录    真实记录   系统矩阵  初始模型  真实模型  
                                                 
    % 绘制反演结果
    t0 = 0;
    t = 1 : 1: length(Model.m);
    figure;
    plot(exp(result(:,iItem))/1000, t, 'r', 'linewidth', 1.5);hold on;
    plot(Model.m/1000, t, 'b--', 'linewidth', 1.5); hold on;
    plot(exp(u0)/1000, t, 'g', 'linewidth', 1.5);   
    set(gcf, 'position', [100 100 230 520]);

    legend('反演结果', '真实模型', '初始模型');
    set(gca, 'xlim', [min(Model.m/1000)-1.5 max(Model.m/1000)+1.5]);
    set(gca, 'ylim', [0 t(end)]);
    set(gca, 'ydir','reverse');
    set(gca,'YTick', 20:40:length(Model.m));
    set(gca,'YTickLabel',1000:500:3000);

%     set(get(gca,'Xlabel'),'String',{'P-Impedence', '(g/cm^3\bullet km/s)'}, 'fontsize', 12);
%     if( iItem == 1)
%         ylabel('Time(ms)', 'fontsize', 12);
%     end
    
    titleName{1} = sprintf('%s-%s-%s-iterNum:%d', typeMethods{item(1)}, ...
        typeObj{item(2)}, typeReg{item(3)}, item(5));
%     titleName{2} = sprintf('lamda:%.3f-sigma:%.4f',item(4), item(6));
%     titleName{2} = sprintf('lamda:%.3f-theta:%.4f',item(4), item(7));
%     title(titleName, 'fontsize', 12); 
    title(tstr{iItem},'fontsize',14, 'position',[9,-3]);
%     title('(b)' ,'fontsize',14, 'position',[9,-3]);
    text(7.4, 205, '波阻抗','fontsize',10);
    text(6.0, 212, '(10^3*g/cc*m/s)','fontsize',10);
%     set(gca,'FontSize',12);
end

%% 稀疏反演
if 1

% 方法选项
sparseMethod = {
    % 字典强度 为0表示完全用字典重建 
    
     % 1代表 迭代  0代表解析解
     %   稀疏度   滑动步长    反演组分    重建组分的大小    迭代次数    是否重建   是否为用迭代的方法求解  
    [ 0     1       1          0.1          2              100         1              1];
     %   稀疏度   滑动步长    字典力度    初始模型力度      迭代次数    是否重建   是否为用迭代的方法求解  
%     [0      1         1          0.1         0.005            8         1               0];
};
global globalParam;
globalParam.convergenceModel = 1;
% sparseItem = size(sparseMethod, 1);
% sparseResult = size(sampNum, sparseItem);
global GSparseInvParam;
nWell = 20;
GSparseInvParam.sizeAtom = 50;              % 原子大小
GSparseInvParam.nAtom = 2500;                % 原子个数
dicSavePath = sprintf('./Dictionaries/DIC_%d_%d_NWell_%d_filt_%.2f.mat', GSparseInvParam.sizeAtom, GSparseInvParam.nAtom, nWell, 1.0);
load(dicSavePath);
GSparseInvParam.DIC = DIC;
GSparseInvParam.xIterNum = 20;

stpShowDictionary(DIC);

maxVal = max(max(abs(Model.d)));
param = sparseMethod{1};
stpInitSparseInvParam_post(param, DIC, sampNum, Model.G/maxVal);
% RMSES = zeros(sparseItem,GSparseInvParam.iterNum );

    if GSparseInvParam.isIterative
        [sparseResult, RMSES] = stpSparseInversion_post(Model.d/maxVal, Model.G/maxVal, u0, log(Model.m));
    else
        [sparseResult, RMSES] = stpSparseInversion_post_fast(Model.d/maxVal, Model.G/maxVal, u0, log(Model.m));
    end


figure;
plot(exp(sparseResult)/1000, t, 'r', 'linewidth', 1.5);hold on;
plot(Model.m/1000, t, 'b--', 'linewidth', 1.5); hold on;
plot(exp(u0)/1000, t, 'g', 'linewidth', 1.5);   
set(gcf, 'position', [100 100 230 520]);

legend('反演结果', '真实模型', '初始模型');
set(gca, 'xlim', [min(Model.m/1000)-1.5 max(Model.m/1000)+1.5]);
set(gca, 'ylim', [0 t(end)]);
set(gca, 'ydir','reverse');
set(gca,'YTick', 20:40:length(Model.m));
set(gca,'YTickLabel',1000:500:3000);

strs{iItem+1} = sprintf('sparseInversion');
end
%% 绘制收敛曲线
if 0
figure;
set(gca,'FontSize',12);
for iItem = 1 : nItems
    plot(res{iItem}, color{iItem}, 'linewidth', 1.5); hold on;
end

legend(strs);
xlabel('迭代次数', 'fontsize', 12); ylabel('均方误差', 'fontsize', 12);
set(gcf, 'position', [100 100 400 300]);
end