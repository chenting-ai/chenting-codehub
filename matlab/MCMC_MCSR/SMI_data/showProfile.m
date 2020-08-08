close all;
%% »­ÆÊÃæ
% load newcolormap;
sampNum=2501;
wellPos=500;
wellAim=z2(wellPos,:)';
upHorizon=zeros(1,991)';
typeName='poumian';
options.basePos = 0;
options.upNum = 0;                    % Êý¾ÝÏÔÊ¾·¶Î§
options.downNum =  sampNum;
options.wellData = wellAim ;                              % ¾®Êý¾Ý
options.wellPos = wellPos;
options.isFilt = 1;
options.firstCDP = 1;
options.xlabelNum = 15;
options.dt = 1;
% load attColor;
% options.colormap = attcolormap;

options.baseTime = upHorizon';              % »ù×¼Ïß
options.timeUp = upHorizon';
options.timeCenter = upHorizon';
options.timeDown = upHorizon';

options.title = typeName;     
options.limits = {'limit', 5800, 8000};
% stpPlotProfile(z2(1:3:end,:)', options );

s_cplot(z2');
% set(gcf, 'position', [100 400 600 450]);
set(gca, 'xlim', [0 991]);
set(gca, 'ylim', [0 2501]);
set(gca,'yTickLabel', 800:50:1050);
set(gca,'xTickLabel', 0:10:100);
caxis([5800,8000]);
hold on;

for i=1:size(wellAim,2)
    for sampling_number=1:size(wellAim,1) 
        if( wellAim(sampling_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:size(wellAim,1) 
        if( wellAim(sampling_number)~=0)
           termination_point=sampling_number;
        end
    end
    y_well=initial_point+40:1:termination_point-20;
    x_well=wellAim(y_well)/10;
    a=mean(x_well);
    x_well=x_well+wellPos-a;
    y_datum=initial_point+30:1:termination_point;
    x_datum=zeros(size(y_datum));
    x_datum(:,:)=wellPos;
    
end
plot(x_datum(1:50),y_datum(1:50),'r--','LineWidth', 1.5) ;hold on;%»­¾®
plot(x_datum(100:150),y_datum(100:150),'r--','LineWidth', 1.5) ;hold on;%»­¾®
plot(x_datum(200:250),y_datum(200:250),'r--','LineWidth', 1.5) ;hold on;%»­¾®
plot(x_datum(300:350),y_datum(300:350),'r--','LineWidth', 1.5) ;hold on;%»­¾®
plot(x_datum(400:450),y_datum(400:450),'r--','LineWidth', 1.5) ;hold on;%»­¾®
plot(x_datum(500:550),y_datum(500:550),'r--','LineWidth', 1.5) ;hold on;%»­¾®
plot(x_well(1:10:end),y_well(1:10:end),'k-','LineWidth', 1.5) ;hold on;%»­¾®


