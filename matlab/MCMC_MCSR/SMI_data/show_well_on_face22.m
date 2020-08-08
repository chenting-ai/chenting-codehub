%% 加载处理数据
T0=20;
markovlen=5;
K1=82;
K2=94;
dicSavePath1 = sprintf('./20200320result/MTDL2_1_50_2000/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52.mat', markovlen, T0);
saveBestX1=load(dicSavePath1);      %%加载结果
name= sprintf('T0_%dmarkovlen%d反演结果过井%d剖面2',T0,markovlen,K1);
path = sprintf('F:/matlab临时图片存储/第五轮重做/MCMC'); 
testsaveeBestX1=saveBestX1. saveBestX(800:1050,:);
% for i=27:32
%     for j=98:105
%         if testsaveeBestX1(j,i)<5880
%             testsaveeBestX1(j,i)=5880;
%         end
%     end
% end
load('E:\matlab_project\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Xline_Inline_number.mat');
trace=Xline_Inline_number(2,K1);
trace2=Xline_Inline_number(2,K2);
load('E:\matlab_project\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('E:\matlab_project\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
well=zeros(2001,2);
for i=1:503
    if Well_timeData(i,K1)==fix(Well_timeData(i,K1))&&Well_timeData(i,K1)~=0
        well(fix(Well_timeData(i,K1)),1)=Well_valueData(i,K1);
    end
end
for i=1:503
    if Well_timeData(i,K2)==fix(Well_timeData(i,K2))&&Well_timeData(i,K2)~=0
        well(fix(Well_timeData(i,K2)),2)=Well_valueData(i,K2);
    end
end
well=well(800:1050,:);
temp_well0=zeros(100,2);
temp_well=zeros(100,2);
%% 将剖面重置
testsaveeBestX11=zeros(100,size(testsaveeBestX1,2));
for k=1:size(testsaveeBestX1,2)
%     if k<40
        testsaveeBestX11(:,k)=testsaveeBestX1(95:194,k);
        if k==trace
            temp_well0(:,:)=well(95:194,:);
        end   
%     elseif k>=40
%         testsaveeBestX11(:,k)=testsaveeBestX1(fix(1.35*k)-4:fix(1.35*k)+86,k);
%         if k==trace
%             temp_well0=well(fix(1.35*k)-4:fix(1.35*k)+86);
%         end
%     end  

end


%%
for sampling_number=1:size(temp_well0,1) 
    if( testsaveeBestX11(sampling_number,trace)~=0)
        initial_point0=sampling_number;
        break;
    end
end
for sampling_number=sampling_number:size(temp_well0,1) 
    if( testsaveeBestX11(sampling_number,trace)~=0)
       termination_point0=sampling_number;
    end
end

temp_well(initial_point0:termination_point0,1)=temp_well0(initial_point0:termination_point0,1);

for sampling_number=1:size(temp_well0,1) 
    if( testsaveeBestX11(sampling_number,trace2)~=0)
        initial_point0=sampling_number;
        break;
    end
end
for sampling_number=sampling_number:size(temp_well0,1) 
    if( testsaveeBestX11(sampling_number,trace2)~=0)
       termination_point0=sampling_number;
    end
end

temp_well(initial_point0:termination_point0,2)=temp_well0(initial_point0:termination_point0,2);

%% 检测有效数据长度
trueMode2=zeros(2,991);
trueModel = temp_well;
temp_x0=1:size(testsaveeBestX11,1);
temp_x02=1:0.1:size(testsaveeBestX11,1);
trueMode2(1,:)=interp1(temp_x0,trueModel(:,1),temp_x02,'linear');
trueMode2(2,:)=interp1(temp_x0,trueModel(:,2),temp_x02,'linear');


temp_x=1:size(testsaveeBestX11,1);
temp_y=1:size(testsaveeBestX11,2);
[temp_x,temp_y]=meshgrid(temp_x,temp_y);
temp_x2=1:0.1:size(testsaveeBestX11,1);
temp_y2=1:0.1:size(testsaveeBestX11,2);
[temp_x2,temp_y2]=meshgrid(temp_x2,temp_y2);
testsaveeBestX_2=interp2(temp_x,temp_y,testsaveeBestX11',temp_x2,temp_y2,'linear');
%%
wellPos=trace*10-5;
wellPos2=trace2*10-5;
wellAim1=trueMode2(1,:)';
wellAim2=trueMode2(2,:)';
for i=1:size(wellAim1,2)
    for sampling_number=1:size(wellAim1,1) 
        if( testsaveeBestX_2(trace*10-5,sampling_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:size(wellAim1,1) 
        if( testsaveeBestX_2(trace*10-5,sampling_number)~=0)
           termination_point=sampling_number;
        end
    end
    y_well=initial_point+15:1:termination_point-30;
    x_well=wellAim1(y_well)/20;
    a=mean(x_well);
    x_well=x_well+wellPos-a;
    y_datum=initial_point+15:1:termination_point-30;
    x_datum=zeros(size(y_datum));
    x_datum(:,:)=wellPos;
    
end

for i=1:size(wellAim2,2)
    for sampling_number=1:size(wellAim2,1) 
        if( testsaveeBestX_2(trace2*10-5,sampling_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:size(wellAim2,1) 
        if( testsaveeBestX_2(trace2*10-5,sampling_number)~=0)
           termination_point=sampling_number;
        end
    end
    y_well2=initial_point+15:1:termination_point-30;
    x_well2=wellAim2(y_well2)/20;
    a=mean(x_well2);
    x_well2=x_well2+wellPos2-a;
    y_datum2=initial_point+15:1:termination_point-30;
    x_datum2=zeros(size(y_datum2));
    x_datum2(:,:)=wellPos2;
    
end
%% 滤波
if 1
    Wn = 0.04;
    testsaveeBestX_2 = csFilterProfile(testsaveeBestX_2, Wn, 'v'); % 纵向滤波
    testsaveeBestX_2 = csFilterProfile(testsaveeBestX_2, Wn, 'h'); % 横向滤波
    
end
%% 画剖面
s_cplot(testsaveeBestX_2');
set(gcf, 'position', [100 500 1100 400]);
% set(gca, 'xlim', [-10 1000]);
set(gca, 'xlim', [0 1411]);
set(gca, 'ylim', [0 994]);
set(gca,'yTickLabel', 895:10:994);
set(gca,'xTickLabel', 0:20:142);
caxis([5800,8000]);
% title('反演剖面');
xlabel('trace Number');
ylabel('Time(ms)');
hold on;


plot(x_datum(1:50),y_datum(1:50),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(100:150),y_datum(100:150),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(200:250),y_datum(200:250),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(300:350),y_datum(300:350),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(400:450),y_datum(400:450),'r-','LineWidth', 1.5) ;hold on;%画井
% plot(x_datum(500:550),y_datum(500:550),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum(500:544),y_datum(500:544),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_well(1:10:end),y_well(1:10:end),'k-','LineWidth', 1.0) ;hold on;%画井


plot(x_datum2(1:50),y_datum2(1:50),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum2(100:150),y_datum2(100:150),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum2(200:250),y_datum2(200:250),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum2(300:350),y_datum2(300:350),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum2(400:450),y_datum2(400:450),'r-','LineWidth', 1.5) ;hold on;%画井
% plot(x_datum(500:550),y_datum(500:550),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_datum2(500:544),y_datum2(500:544),'r-','LineWidth', 1.5) ;hold on;%画井
plot(x_well2(10:10:end),y_well2(10:10:end),'k-','LineWidth', 1.0) ;hold on;%画井
% fileName = sprintf('%s/%s.tif', path, name);
% print('-dtiff', '-r600', fileName);