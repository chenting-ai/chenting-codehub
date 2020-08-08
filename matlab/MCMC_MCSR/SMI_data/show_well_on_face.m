%% 加载处理数据
T0=20;
markovlen=5;
K1=82;
dicSavePath1 = sprintf('./ResultsData52_nohorizon_move/MCMC/markovlen%d_T0%fsaveBestX1.mat', markovlen, T0);
saveBestX1=load(dicSavePath1);      %%加载结果
name= sprintf('T0_%dmarkovlen%dMCMC52反演结果过井%d剖面',T0,markovlen,K1);
path = sprintf('F:/matlab临时图片存储/第五轮/MCMC'); 
testsaveeBestX1=saveBestX1. saveBestX(800:1050,1:100);
for i=27:32
    for j=98:105
        if testsaveeBestX1(j,i)<5850
            testsaveeBestX1(j,i)=5850;
        end
    end
end
load('.\Load_WellData\Xline_Inline_number.mat');
trace=Xline_Inline_number(1,K1);
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
well=zeros(2001,1);
temp_well=zeros(2001,1);
for i=1:503
    if Well_timeData(i,K1)==fix(Well_timeData(i,K1))&&Well_timeData(i,K1)~=0
        well(fix(Well_timeData(i,K1)),1)=Well_valueData(i,K1);
    end
end
well=well(800:1050);
temp_well=temp_well(800:1050);
for sampling_number=1:size(well,1) 
    if( testsaveeBestX1(sampling_number,trace)~=0)
        initial_point0=sampling_number;
        break;
    end
end
for sampling_number=sampling_number:size(well,1) 
    if( testsaveeBestX1(sampling_number,trace)~=0)
       termination_point0=sampling_number;
    end
end
temp_well(initial_point0:termination_point0)=well(initial_point0:termination_point0);


%% 检测有效数据长度

trueModel = temp_well;
temp_x=1:size(testsaveeBestX1,1);
temp_x2=1:0.1:size(testsaveeBestX1,1);
trueMode2=interp1(temp_x,trueModel',temp_x2,'linear');



temp_x=1:size(testsaveeBestX1,1);
temp_y=1:size(testsaveeBestX1,2);
[temp_x,temp_y]=meshgrid(temp_x,temp_y);
temp_x2=1:0.1:size(testsaveeBestX1,1);
temp_y2=1:0.1:size(testsaveeBestX1,2);
[temp_x2,temp_y2]=meshgrid(temp_x2,temp_y2);
testsaveeBestX_2=interp2(temp_x,temp_y,testsaveeBestX1',temp_x2,temp_y2,'linear');


%% 画剖面
wellPos=trace*10-5;
wellAim=trueMode2';
s_cplot(testsaveeBestX_2');
set(gcf, 'position', [100 500 600 450]);
set(gca, 'xlim', [-10 1000]);
set(gca, 'ylim', [0 2501]);
set(gca,'yTickLabel', 800:50:1050);
set(gca,'xTickLabel', 0:10:100);
caxis([5800,8000]);
% title('反演剖面');
xlabel('trace Number');
ylabel('Time(ms)');
hold on;

for i=1:size(wellAim,2)
    for sampling_number=1:size(wellAim,1) 
        if( testsaveeBestX_2(trace*10-5,sampling_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:size(wellAim,1) 
        if( testsaveeBestX_2(trace*10-5,sampling_number)~=0)
           termination_point=sampling_number;
        end
    end
    y_well=initial_point+30:1:termination_point-30;
    x_well=wellAim(y_well)/20;
    a=mean(x_well);
    x_well=x_well+wellPos-a;
    y_datum=initial_point+30:1:termination_point-30;
    x_datum=zeros(size(y_datum));
    x_datum(:,:)=wellPos;
    
end
plot(x_datum(1:50),y_datum(1:50),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(100:150),y_datum(100:150),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(200:250),y_datum(200:250),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(300:350),y_datum(300:350),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(400:450),y_datum(400:450),'r-','LineWidth', 1.5) ;hold on;%画井
% plot(x_datum(500:550),y_datum(500:550),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_well(1:10:end),y_well(1:10:end),'k-','LineWidth', 1.0) ;hold on;%画井

mkdir(sprintf('%s/tif_acrosswell', path));
fileName = sprintf('%s/tif_acrosswell/%s.tif', path, name);
print('-dtiff', '-r600', fileName);
