% ����һάģ�ͷ���
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
close all;

%% ����ģ������
load('./Load_WellData/marmousi_imp_horizon_350_377.mat');

% ����
waveletFreq = 30;
wave = s_create_wavelet({'type','ricker'}, {'frequencies',waveletFreq}, {'step', 1}, {'wlength',200});
wavelet = wave.traces;
% si = 160;
si = 1;
trueModel = trueModel(si:end, :);
sampNum = size(trueModel, 1);
[poststack, G] = csGenPost1DForwardModel(wavelet, trueModel, sampNum, sampNum-1 );
% �������
noiseType = 1;
noiseDB = 4;
noiseOption = {'gs','lp'};
postNoise = wyjGenNormNoise2D(poststack, noiseDB, noiseType);
% ȡ��
traceId = 11;
Model.G = G; % ���ݾ��� 
Model.d0 = poststack(:, traceId);
Model.m = trueModel(:, traceId);
Model.d = postNoise(:, traceId);
% 
% modelName = ['model-' noiseOption{noiseType} '-' sprintf('%d', noiseDB) 'DB.mat'];
% save([ './data/' modelName],  'Model');
% snr = SNR_singlech(Model.d0, Model.d);
%% ����ģ��

% load('./data/model-gs-6DB.mat');
[sampNum, ~] = size(Model.m);
% Model.d = Model.d0;
% ��ʼģ�� 
Wn = 0.03;
initModel = csFilterProfile(trueModel, Wn, 'v'); % �����˲�
initModel = csFilterProfile(initModel, Wn, 'h'); % �����˲�
u0 = log(initModel(:, traceId));
% ������� , Ҳ����ʹ��model�����е�������¼
% [postNoise ,Noise_lp_6]= wyjGenNormNoise(Model.d0, 6, 2); % ������ʦ�ĺ���
% [postNoise ,Noise_lp_6]= stpGenNormNoise(Model.d0, 6, 2); % ���Ǳ�ĺ���
%% ���ò���
% ���������Ϣ
typeMethods = {'GD', 'BFGS', 'CG', 'NCG', 'VSSNSGA', 'csNCG'};  % ��������(1-�ݶ��½� 2-��ţ�ٷ� 3-�����ݶȷ� 4-�Ǳ��⹲���ݶ� 5-�����ݶȷ� 6-����Ӧ�⹲��)
typeObj = {'L1', 'L2', 'huber', 'MCC','MCCi', 'SAM', 'GEV'};  % Ŀ�꺯��(1-L1,2-L2,3-huber,4-��д��Mcc,5-�Ǳ�д��Mcci,6-��Ϸ���,7-���弫ֵ����)    
typeReg = {'none', 'Tic1', 'TV', 'MGS'};
testItems = {
 % �����㷨��Ŀ�꺯������������򻯲���lamda,����������MGS��sigmaֵ��MCC��thetaֵ

% 6db ��˹���� ������
%     [3 2 1 0  30     0.0001 0.001];

% 6db ��˹���� tikhonov����
%     [3 2 2 0.05  30     0.0001 0.001];
%     [3 2 2 0.5  30     0.0001 0.001];
%     [3 2 2 5 80     0.0001 0.001];
% 6db ��˹���� ���򻯶Ա�

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

%% ������ʵģ��
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

% text(7.4, 205, '���迹','fontsize',10);
% text(6.0, 212, '(10^3*g/cc*m/s)','fontsize',10);
% text(1.8, 116, 'Time(ms)','fontsize',12,  'rotation', 90);

text(7.2, 375, '���迹','fontsize',10);
text(5.8, 388, '(10^3*g/cc*m/s)','fontsize',10);
text(0.2, 216, 'Time(ms)','fontsize',12,  'rotation', 90);

% title('(a)','fontsize',14, 'position',[9,-3]);
title('(a)','fontsize',14, 'position',[8.2,-3]);
% 2
t2 = 1 : 1: length(Model.d);
subplot(1, 2 ,2);
plot(Model.d0, t2, 'b', 'linewidth',linewidth);hold on;
plot(Model.d, t2, 'r--', 'linewidth', linewidth);
legend('������','������');

set(gca, 'xlim', [-0.6 0.6]);

set(gca, 'ylim', [0 t2(end)]);
set(gca, 'ylim', [0 t1(end)]);
set(gca, 'ydir','reverse');
set(gca,'YTick', 20:40:length(Model.d));
set(gca,'YTickLabel',1000:500:3000);

% text(-0.2, 205, '���','fontsize',10);
text(-0.1, 375, '���','fontsize',10);

title('(b)','fontsize',14, 'position',[-0.0,-3]);
end
%% ��ʼ����
strs = cell(nItems+1, 1);
result = zeros(sampNum, nItems);
res = cell(1, nItems);
for iItem = 1 : nItems
    item = testItems{iItem};
    strs{iItem} = sprintf('%s-%s-%s-lamda:%.3f-sigma:%.4f', typeMethods{item(1)}, ...
        typeObj{item(2)}, typeReg{item(3)}, item(4), item(6));
    % ���
%     [result(:,iItem), res{iItem}] = csLinearSolver(Model.d, Model.d0, Model.G, u0, log(Model.m), item);       
    [result(:,iItem), res{iItem}] = csLinearSolver(Model.d, [], Model.G, u0, [], item);       
                                             %    ������¼    ��ʵ��¼   ϵͳ����  ��ʼģ��  ��ʵģ��  
                                                 
    % ���Ʒ��ݽ��
    t0 = 0;
    t = 1 : 1: length(Model.m);
    figure;
    plot(exp(result(:,iItem))/1000, t, 'r', 'linewidth', 1.5);hold on;
    plot(Model.m/1000, t, 'b--', 'linewidth', 1.5); hold on;
    plot(exp(u0)/1000, t, 'g', 'linewidth', 1.5);   
    set(gcf, 'position', [100 100 230 520]);

    legend('���ݽ��', '��ʵģ��', '��ʼģ��');
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
    text(7.4, 205, '���迹','fontsize',10);
    text(6.0, 212, '(10^3*g/cc*m/s)','fontsize',10);
%     set(gca,'FontSize',12);
end

%% ϡ�跴��
if 1

% ����ѡ��
sparseMethod = {
    % �ֵ�ǿ�� Ϊ0��ʾ��ȫ���ֵ��ؽ� 
    
     % 1���� ����  0���������
     %   ϡ���   ��������    �������    �ؽ���ֵĴ�С    ��������    �Ƿ��ؽ�   �Ƿ�Ϊ�õ����ķ������  
    [ 0     1       1          0.1          2              100         1              1];
     %   ϡ���   ��������    �ֵ�����    ��ʼģ������      ��������    �Ƿ��ؽ�   �Ƿ�Ϊ�õ����ķ������  
%     [0      1         1          0.1         0.005            8         1               0];
};
global globalParam;
globalParam.convergenceModel = 1;
% sparseItem = size(sparseMethod, 1);
% sparseResult = size(sampNum, sparseItem);
global GSparseInvParam;
nWell = 20;
GSparseInvParam.sizeAtom = 50;              % ԭ�Ӵ�С
GSparseInvParam.nAtom = 2500;                % ԭ�Ӹ���
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

legend('���ݽ��', '��ʵģ��', '��ʼģ��');
set(gca, 'xlim', [min(Model.m/1000)-1.5 max(Model.m/1000)+1.5]);
set(gca, 'ylim', [0 t(end)]);
set(gca, 'ydir','reverse');
set(gca,'YTick', 20:40:length(Model.m));
set(gca,'YTickLabel',1000:500:3000);

strs{iItem+1} = sprintf('sparseInversion');
end
%% ������������
if 0
figure;
set(gca,'FontSize',12);
for iItem = 1 : nItems
    plot(res{iItem}, color{iItem}, 'linewidth', 1.5); hold on;
end

legend(strs);
xlabel('��������', 'fontsize', 12); ylabel('�������', 'fontsize', 12);
set(gcf, 'position', [100 100 400 300]);
end