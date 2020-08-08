%% 加载处理数据
T0=20;
markovlen=5;
dicSavePath1 = sprintf('./ResultsData52_horizon_move/MTDL2_1_25_2340/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52horizon25_2340.mat', markovlen, T0);
saveBestX1=load(dicSavePath1);      %%加载结果
testsaveeBestX1=saveBestX1. saveBestX(800:1050,1:100);
for i=27:32
    for j=98:105
        if testsaveeBestX1(j,i)<5850
            testsaveeBestX1(j,i)=5850;
        end
    end
end
%% 加载地震剖面
seismic_data=load('.\SeismicData\inline99_seismic.txt');  
seismic_data=seismic_data(800:1050,1:100);
%% 将剖面重置
testsaveeBestX11=zeros(91,size(testsaveeBestX1,2));
seismic_data11=zeros(91,size(testsaveeBestX1,2));
for k=1:size(testsaveeBestX1,2)
    if k<40
        testsaveeBestX11(:,k)=testsaveeBestX1(k+10:k+100,k);
        seismic_data11(:,k)=seismic_data(k+10:k+100,k);
    elseif k>=40
        testsaveeBestX11(:,k)=testsaveeBestX1(fix(1.35*k)-4:fix(1.35*k)+86,k);
        seismic_data11(:,k)=seismic_data(fix(1.35*k)-4:fix(1.35*k)+86,k);
    end  
end






%% 检测有效数据长度
seismic_data2=zeros(901,100);
for trace_number=1:size(testsaveeBestX1,2)
    temp_x0=1:size(testsaveeBestX11,1);
    temp_x02=1:0.1:size(testsaveeBestX11,1);
    seismic_data2(:,trace_number)=interp1(temp_x0,seismic_data11(:,trace_number)',temp_x02,'linear');
end
temp_x=1:size(testsaveeBestX11,1);
temp_y=1:size(testsaveeBestX11,2);
[temp_x,temp_y]=meshgrid(temp_x,temp_y);
temp_x2=1:0.1:size(testsaveeBestX11,1);
temp_y2=1:0.1:size(testsaveeBestX11,2);
[temp_x2,temp_y2]=meshgrid(temp_x2,temp_y2);
testsaveeBestX_2=interp2(temp_x,temp_y,testsaveeBestX11',temp_x2,temp_y2,'linear');

%% 画剖面
s_cplot(testsaveeBestX_2');
set(gcf, 'position', [100 500 500 200]);
set(gca, 'xlim', [-10 1000]);
set(gca, 'ylim', [0 900]);
set(gca,'yTickLabel', 810:20:900);
set(gca,'xTickLabel', 0:20:100);
caxis([5800,8000]);
% title('反演剖面');
xlabel('trace Number');
ylabel('Time(ms)');
hold on;

if 0
    for trace=1:size(seismic_data2,2)
    for sampling_number=1:size(seismic_data2,1)
        if( testsaveeBestX_2(trace*10-5,sampling_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:size(seismic_data2,1)
        if( testsaveeBestX_2(trace*10-5,sampling_number)~=0)
           termination_point=sampling_number;
        end
    end
    Pos=trace*10-5;
%     y_well=initial_point+15:1:termination_point-30;
    y=1:size(seismic_data2,1);
    x=seismic_data2(y,trace)/1500;
    a=mean(x(initial_point:termination_point));
    point=Pos-a;
    x=x+point;
%     for sample=1:size(seismic_data2,1)
%         if x(sample,1)>point
%             plot(point:0.5:x(sample,1),y(1,sample):length(point:0.5:x(sample,1)):y(1,sample),'k-','LineWidth', 0.5) ;hold on;%
%         else
%             plot(x(sample,1),y(1,sample),'k-','LineWidth', 0.5) ;hold on;%
%         end
%         
%     end
    
    plot(x(1:10:end),y(1:10:end),'k-','LineWidth', 0.5) ;hold on;%
    
end
end
name= sprintf('T0_%dmarkovlen%dMCMC52反演结果过井%d剖面',T0,markovlen,K1);
path = sprintf('F:/matlab临时图片存储/第五轮/MCMC'); 
% mkdir(sprintf('%s/tif_acrosswell', path));
% fileName = sprintf('%s/tif_acrosswell/%s.tif', path, name);
% print('-dtiff', '-r600', fileName);
if 0
    figure('color',[1 1 1]);
    set(gcf, 'position', [100 500 500 200]);
    set(gca, 'ylim', [0 900]);
    set(gca,'yTickLabel', 810:20:900);
    set(gca,'xTickLabel', 0:10:100);
    xlabel('trace Number');
    ylabel('Time(ms)');
    set(gca,'xaxislocation','top');
    wigb(seismic_data11);
end