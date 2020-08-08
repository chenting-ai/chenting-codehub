%%%%画图
T0=20;
markovlen=5;
dicSavePath1 = sprintf('./ResultsData52_horizon_move/MTDL2_1_25_2340/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52horizon25_2340.mat', markovlen, T0);
saveBestX1=load(dicSavePath1);      %%加载结果

name= sprintf('T0_%dmarkovlen%dMCDL52反演结果25_2340',T0,markovlen);
path = sprintf('F:/matlab临时图片存储/第五轮/MCDL_2'); 
testsaveeBestX1=saveBestX1. saveBestX(800:1050,1:100);

x=1:size(testsaveeBestX1,1);
y=1:size(testsaveeBestX1,2);
[x,y]=meshgrid(x,y);
x2=1:0.1:size(testsaveeBestX1,1);
y2=1:0.1:size(testsaveeBestX1,2);
[x2,y2]=meshgrid(x2,y2);
z2=interp2(x,y,testsaveeBestX1',x2,y2,'linear');

figure('color',[1 1 1])
set(gcf, 'position', [100 500 600 450]);
mesh(x2,y2,z2);%%反演波阻抗
view(90,90)  
box off
axis off
grid off
set(gca, 'ylim', [1 100]);
set(gca, 'xlim', [0 250]);
set(gca,'xTickLabel', 800:50:1050);
shading interp;
colorbar
caxis([5800,8000]);
% mkdir(sprintf('%s/tif', path));
% fileName = sprintf('%s/tif/%s.tif', path, name);
% print('-dtiff', '-r600', fileName);

% 
% figure
% set(gcf, 'position', [200 200 600 450]);
% imagesc(testsaveeBestX1);
% set(gca, 'ylim', [0 250]);
% set(gca, 'xlim', [0 100]);
% set(gca,'YTickLabel', 800:50:1050);
% colorbar
% caxis([5500,8000]);
% % 




%% 画剖面
% % load newcolormap;
% % close all;
% options.basePos = 0;
% options.upNum = 0;                    % 数据显示范围
% options.downNum =  sampNum;
% options.wellData = [] ;                              % 井数据
% options.wellPos = [];
% options.isFilt = 1;
% options.firstCDP = 1;
% options.xlabelNum = 15;
% options.dt = 1;
% load('./original_color.mat');
% options.colormap = original_color;
% 
% options.baseTime = upHorizon';              % 基准线
% options.timeUp = upHorizon';
% options.timeCenter = upHorizon';
% options.timeDown = upHorizon';
% 
% p = 5 * 2;
% options.title = [];     
% options.limits = {'limit', 5800, 8000};
% stpPlotProfile(testsaveeBestX1, options );
% set(gcf, 'position', [200 100 600 450]);
% shading interp;