clear
clc

%% 加载数据
true_impedance=load('./result/MCMC/markovlen15_T0100.000000saveBestX1_inlimit_2.mat');
true_impedance=true_impedance.saveBestX;
% true_impedance=load('./usefuldata/marmousi2_SM.mat');
% true_impedance=true_impedance.marmousi2_SM;

true_impedance=medfilt2(true_impedance,[2,7]);

% Wn = 0.5;
% true_impedance = csFilterProfile(true_impedance, Wn, 'v'); % 纵向滤波
% true_impedance = csFilterProfile(true_impedance, Wn, 'h'); % 横向滤波

%% 插值
temp_x=1:size(true_impedance,1);
temp_y=1:size(true_impedance,2);
[temp_x,temp_y]=meshgrid(temp_x,temp_y);
temp_x2=1:0.1:size(true_impedance,1);
temp_y2=1:0.1:size(true_impedance,2);
[temp_x2,temp_y2]=meshgrid(temp_x2,temp_y2);
true_impedance2=interp2(temp_x,temp_y,true_impedance',temp_x2,temp_y2,'linear');

%% 画剖面
s_cplot(true_impedance2')
set(gcf, 'position', [200 300 600 400]);
% set(gca, 'xlim', [-10 1000]);
set(gca, 'xlim', [0 size(true_impedance2',2)]);
set(gca, 'ylim', [0 size(true_impedance2',1)]);
set(gca,'yTickLabel', 0:20:200);
set(gca,'xTickLabel', 0:50:351);

caxis([5000,9500]);
% title('反演剖面');
xlabel('trace Number');
ylabel('Time(ms)');
hold on;



% 
% 
% 
% 
% 
% 
