%% 利用SMI方法做云景数据
% 2020/2/16
% 陈挺
clc;clear;
num = 9; % 用来插值的井数量
DPS = 4;%Den5，P2，S3，G1，VP/Vs4,seismic6
isfilter=1;%滤波否
%% 读取井数据和地震数据
for k=1
    % 地震
    seismic_data=load('.\LNusefuldata\cdp_stack_3600_4000newT1_50_1001ms.mat');    
    seismic_data=seismic_data.cdp_stack_3600_4000newT1_50_1001ms;
    point_number = size(seismic_data,1);
    trace_number = size(seismic_data,2);
    % 井数据
    load('.\LNusefuldata\well_data_newhor.mat');%加载测井值
    well_data=well_data_newhor;
    wellpoint = size(well_data,1); 
    near_well_seismic_data = well_data(:,6,:);
    well_number = size(well_data,3);
    % 对井数据进行滤波
    if 1
        Wn=0.3;
        for i=1:well_number
            well_data(:,DPS,i)= csFilterProfile(well_data(:,DPS,i), Wn, 'v');   % 纵向滤波  
        end 
    end
    
    

    
end
%% 开始流程

temp_Impedence_data=zeros(wellpoint,trace_number);
% h=waitbar(0,'please wait');
CoreNum=3; %the number of cpu 5
if matlabpool('size')<=0  
    matlabpool('open','local',CoreNum);
else  
    disp('matlab pool already started');
end
parfor i=1:trace_number
%     str=['运行中...',num2str(i/trace_number*100),'%'];
%     waitbar(i/trace_number,h,str)
    i
    % 提取一道地震数据的有效数据段  seismic_val
    up=1;
    down=1;
    for j=1:point_number
        if seismic_data(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up:point_number
        if seismic_data(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    if useful_len<10
       continue; 
    end
    seismic_val=seismic_data(up:down,i);
    % 插值，将地震数据插值成井数据一样的点数 
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint);
    seismic_val=interp1(useful_len_seismic,seismic_val,inter_useful_len_seismic,'spline');
    %计算相关系数CORcoef,需要做归一化再算相关
    CORcoef = zeros(1,well_number);
    for K1 = 1:well_number
        seismic_val = mapminmax(seismic_val);
        seismic_val = seismic_val';
        simpleseismic = near_well_seismic_data(:,K1);
        simpleseismic = simpleseismic';
        simpleseismic = mapminmax(simpleseismic);
        simpleseismic = simpleseismic';
        CORcoef(1,K1) = min(min(corrcoef(simpleseismic,seismic_val)));
    end

    % 利用相关系数较高的几口井和相关系数对地震数据位置处的波阻抗进行插值
    % 找位置     存在问题：相关系数大多是负数。
    absCORcoef=abs(CORcoef);
    val = zeros(num,1);
    location = zeros(num,1);
    for k = 1:num
        [val(k,1),location(k,1)] = max(absCORcoef);
        absCORcoef(1,location(k,1)) = 0;
    end
    
    % 插值
    if 1 %将相关系数的分布从新归置
        maxval = max(max(val));
        minval = min(min(val));
        max_minlen = maxval-minval;
        for kk = 1:num
            val(kk,1) = (val(kk,1)-minval)/max_minlen;
        end 
    end
    %利用相关系数插值
    valsum = sum(val);
    inter_impedence = zeros(wellpoint,1);
    for k = 1:num
        if CORcoef(1,location(k,1))>0
            val(k,1) = val(k,1)/valsum;
        else
            val(k,1) = val(k,1)/valsum;
        end
        inter_impedence = inter_impedence+val(k,1)*well_data(:,DPS,location(k,1));
    end
    temp_Impedence_data(:,i)=inter_impedence;
    
end

%% 写成segy数据
temp_Impedence_data2=zeros(size(temp_Impedence_data));
Impedence_data=zeros(size(seismic_data));
%%%%%%%%%%%%%导入地震数据信息%%%%%%%%%%%%%%
[seismic,bic_header,binary_header]=read_segy_file('./20200310/cdp_stack_3600_4000newT1-50+1001ms.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %读入地震数据


if isfilter 
    Wn=0.2;
    temp_Impedence_data2 = csFilterProfile(temp_Impedence_data, Wn, 'h');   % 纵向滤波  
end

for i=1:trace_number
    up=1;
    down=1;
    for j=1:point_number
        if seismic_data(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up:point_number
        if seismic_data(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint);
    useful_len_seismic = linspace(1,useful_len,useful_len);
    %将结果插值成原来的点数
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %依次循环完每一道
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0320T1_T31ms_one_well8_filter0.2_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

if isfilter 
    Wn=0.4;
    temp_Impedence_data2 = csFilterProfile(temp_Impedence_data, Wn, 'h');   % 纵向滤波  
end
for i=1:trace_number
    up=1;
    down=1;
    for j=1:point_number
        if seismic_data(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up:point_number
        if seismic_data(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint);
    useful_len_seismic = linspace(1,useful_len,useful_len);
    %将结果插值成原来的点数
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %依次循环完每一道
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0320T1_T31ms_one_well8_filter0.4_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据






%% 将地震数据的频带加到插值好的波阻抗频带