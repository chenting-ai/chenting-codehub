%% 绘制单道反演结果
%% 加载数据
clear
clc
T0=100;
markovlen=15;
K1=201;
public_path='E:\matlab_project\The_stochastic_inversion\NewDictionary_MCMC_inversion_move4\marmousi_SM\';
truemodel_path=[public_path,'usefuldata\marmousi2_SM.mat'];%marmousi2_SM
load(truemodel_path);
trueModel=marmousi2_SM(:,K1);

MCMC_path = sprintf('%sresult/MCMC/markovlen%d_T0%fsaveBestX1trace201_inlimit_2.mat',public_path, markovlen, T0);
MCMC=load(MCMC_path);
MCMC=MCMC.saveBestX(:,K1);

MCDL_path = sprintf('%sresult/MCDL/markovlen%d_T0%fsaveBestX1trace201_inlimit_2_2.mat',public_path, markovlen, T0);
MCDL=load(MCDL_path);
MCDL=MCDL.saveBestX(:,K1);
uesful_len=size(trueModel,1);

t =1:uesful_len;
linewidth = 1.5;
figure('color',[1 1 1]);set(gcf, 'position', [200 200 300 700]);
plot(trueModel/1000, t , 'g', 'linewidth', linewidth);hold on;
plot(MCMC/1000, t , 'r--', 'linewidth', linewidth);hold on;
plot(MCDL/1000, t , 'k', 'linewidth', linewidth);hold on;
legend({'tureModel', 'MCMC', 'MCSR'});
xlabel('Impedance(10^3*g/cc*m/s)');
ylabel('Time(ms)');
set(gca, 'ydir','reverse');
a1=corr(trueModel,MCMC);
a2=sum((trueModel-MCMC).*(trueModel-MCMC))/(uesful_len);
b1=corr(trueModel,MCDL);
b2=sum((trueModel-MCDL).*(trueModel-MCDL))/(uesful_len);

fprintf('MCMC COR:%f  MSE:%f \nMCDL COR:%f  MSE:%f\n',a1,a2,b1,b2);
sprintf('markovlen%d_T0%ftrace201MCMCCOR%fMSE%f_MCDLCOR%fMSE%f',markovlen, T0,a1,a2,b1,b2)
%% 
% inital_path=[public_path,'usefuldata\marmousi_SM_init_0.1.mat'];
% inital=load(inital_path);
% inital=inital.marmousi_SM_init(:,201);
% c1=corr(trueModel,inital);
% c2=sum((trueModel-inital).*(trueModel-inital))/(uesful_len);
% fprintf('inital COR:%f  MSE:%f \n',c1,c2);
% figure('color',[1 1 1]);set(gcf, 'position', [200 200 300 700]);
% plot(trueModel/1000, t , 'g', 'linewidth', linewidth);hold on;
% plot(saveBestX1/1000, t , 'r--', 'linewidth', linewidth);hold on;
% plot(saveBestX2/1000, t , 'k', 'linewidth', linewidth);hold on;
% % plot(saveBestX3/1000, t , 'k', 'linewidth', linewidth);hold on;
% % plot(horizon/1000, t , 'r', 'linewidth', linewidth);hold on;
% set(gca, 'xlim', [4.5 9.5]);
% set(gca,'YTick', 1:10:useful_len);
% set(gca,'YTickLabel', 905:10:964);
% xlabel('Impedance(10^3*g/cc*m/s)');
% set(gca, 'ydir','reverse');
% ylabel('Time(ms)');
% legend({'Well', 'MCMC', 'MCDL'});
% title('W3-2104');
% a1=corr(trueModel,saveBestX1);
% a2=sum((trueModel-saveBestX1).*(trueModel-saveBestX1))/(useful_len);
% b1=corr(trueModel,saveBestX2);
% b2=sum((trueModel-saveBestX2).*(trueModel-saveBestX2))/(useful_len);
name= sprintf('T0_%dmarkovlen%d反演结果井%dMCDLcor%f_MSE%f_MCMCcor%f_MSE%f',T0,markovlen,K1,b1,b2,a1,a2);
% % path = sprintf('F:/matlab临时图片存储/0323论文用图/MCDL_2_50_2000'); 
% fileName = sprintf('%s/%s.tif', path, name);
% print('-dtiff', '-r600', fileName);