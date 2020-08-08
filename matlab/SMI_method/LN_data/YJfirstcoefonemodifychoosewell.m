%% 实现SMI方法
% 2020/1/10
% 陈挺
clc;clear;
coefval = 0.8; %相关系数阈值
tempnum = 11; % 如果一口井都没选出来，就用前num口井来插值
DPS = 4;%Den5，P2，S3，G1，VP/Vs4,seismic6
resample_point1 = 100; % 第一层重采样点数
resample_point2 = 100; % 第二层重采样点数
isfilter = 1;
h=waitbar(0,'please wait');
% CoreNum=3; %the number of cpu 5
% if matlabpool('size')<=0  
%     matlabpool('open','local',CoreNum);
% else  
%     disp('matlab pool already started');
% end
%% 读取井数据和地震数据 线道号 层位
for k=1
    % 地震
    seismic_data1=load('.\LNusefuldata\cdp_stack1500offset1msT1_T2smooth.mat');    
    seismic_data1=seismic_data1.cdp_stack1500offset1msT1_T2smooth;
    point_number1 = size(seismic_data1,1);
    trace_number = size(seismic_data1,2);
    % 井数据
    load('.\LNusefuldata\welldataT1_T2_2.mat');%加载测井值
    well_data1=welldataT1_T2;
    wellpoint1 = size(well_data1,1); 
    near_well_seismic_data1 = well_data1(:,6,:);
    well_number1 = size(well_data1,3);
    % 对井数据进行滤波
    if 1
        Wn=0.4;
        for i=1:well_number1
            well_data1(:,DPS,i)= csFilterProfile(well_data1(:,DPS,i), Wn, 'v');   % 纵向滤波  
        end 
    end

    
    
end
%% 开始流程

temp_Impedence_data1=zeros(wellpoint1,trace_number);

for i=1:trace_number
    str=['1运行中...',num2str(i/trace_number*100),'%'];
    waitbar(i/trace_number,h,str)
%     i
    % 提取一道地震数据的有效数据段  seismic_val
    up=1;
    down=1;
    for j=1:point_number1
        if seismic_data1(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up:point_number1
        if seismic_data1(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    if useful_len<10
       continue; 
    end
    seismic_val=seismic_data1(up:down,i);
    all_point = size(seismic_data1,1);
    % 插值，将地震数据插值成井数据一样的点数 
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint1);
    seismic_val=interp1(useful_len_seismic,seismic_val,inter_useful_len_seismic,'spline');

    %% 优选n口井出来，只用这n口井
    % 提取一道地震数据的有效数据段  seismic_val与所有井数据算相关系数
    tempCORcoef = zeros(1,well_number1);
    seismic_val = mapminmax(seismic_val);
    seismic_val = seismic_val';
    for K1 = 1:well_number1
        simpleseismic = near_well_seismic_data1(:,K1);
        simpleseismic = simpleseismic';
        simpleseismic = mapminmax(simpleseismic);
        simpleseismic = simpleseismic';
        tempCORcoef(1,K1) = min(min(corrcoef(simpleseismic,seismic_val)));
    end

    % 优选n口井.利用相关系数阈值
    tempabsCORcoef=abs(tempCORcoef);
    location=[];
    val=[];
    num = 0;%选出的井数量
    for k = 1:well_number1
        if tempabsCORcoef(1,k)>coefval
            location=[location;k];
            val=[val;tempabsCORcoef(1,k)];
            num = num+1;
        end
    end
    
    if num==0
        num = tempnum;
        temp_well_data = zeros(resample_point1,5,num);
        temp_seismic_data = zeros(resample_point1,num);
        val = zeros(num,1);
        location = zeros(num,1);
        for k = 1:num
            [val(k,1),location(k,1)] = max(tempabsCORcoef);
            temp_well_data(:,:,k) = well_data1(:,1:5,location(k,1));
            temp_seismic_data(:,k) = near_well_seismic_data1(:,location(k,1));
            tempabsCORcoef(1,location(k,1)) = 0;
        end
    else
        temp_well_data = zeros(resample_point1,5,num);
        temp_seismic_data = zeros(resample_point1,num);
        for k = 1:num
            temp_well_data(:,:,k) = well_data1(:,1:5,location(k,1));
            temp_seismic_data(:,k) = near_well_seismic_data1(:,location(k,1));
        end
    end
     
    if num>=2 %将相关系数的分布从新归置
        maxval = max(max(val));
        minval = min(min(val));
        max_minlen = maxval-minval;
        for kk = 1:num
            val(kk,1) = (val(kk,1)-minval)/max_minlen;
        end 
    else
        val(:)=1;
    end
   
    %利用相关系数插值
    valsum = sum(val);
    inter_impedence = zeros(wellpoint1,1);
    for k = 1:num
        val(k,1) = val(k,1)/valsum;
        inter_impedence = inter_impedence+val(k,1)*well_data1(:,DPS,location(k,1));
    end
    temp_Impedence_data1(:,i)=inter_impedence;
    
end
% 

for k=1
    % 地震
    seismic_data2=load('.\LNusefuldata\cdp_stack1500offset1msT2_T3smooth.mat');    
    seismic_data2=seismic_data2.cdp_stack1500offset1msT2_T3smooth;
    point_number2 = size(seismic_data2,1);
    trace_number = size(seismic_data2,2);
    % 井数据
    load('.\LNusefuldata\welldataT2_T3_2.mat');%加载测井值
    well_data2=welldataT2_T3;
    wellpoint2 = size(well_data2,1); 
    near_well_seismic_data2 = well_data2(:,6,:);
    well_number2 = size(well_data2,3);
    % 对井数据进行滤波
    if 1
        Wn=0.4;
        for i=1:well_number2
            well_data2(:,DPS,i)= csFilterProfile(well_data2(:,DPS,i), Wn, 'v');   % 纵向滤波  
        end 
    end

    
    
end
%% 开始流程
temp_Impedence_data2=zeros(wellpoint2,trace_number);
for i=1:trace_number
    str=['2运行中...',num2str(i/trace_number*100),'%'];
    waitbar(i/trace_number,h,str)
%     i
    % 提取一道地震数据的有效数据段  seismic_val
    up=1;
    down=1;
    for j=1:point_number2
        if seismic_data2(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up:point_number2
        if seismic_data2(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    if useful_len<10
       continue; 
    end
    seismic_val=seismic_data2(up:down,i);
    all_point = size(seismic_data2,1);
    % 插值，将地震数据插值成井数据一样的点数 
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint2);
    seismic_val=interp1(useful_len_seismic,seismic_val,inter_useful_len_seismic,'spline');

    %% 优选n口井出来，只用这n口井
    % 提取一道地震数据的有效数据段  seismic_val与所有井数据算相关系数
    tempCORcoef = zeros(1,well_number2);
    seismic_val = mapminmax(seismic_val);
    seismic_val = seismic_val';
    for K1 = 1:well_number2
        simpleseismic = near_well_seismic_data2(:,K1);
        simpleseismic = simpleseismic';
        simpleseismic = mapminmax(simpleseismic);
        simpleseismic = simpleseismic';
        tempCORcoef(1,K1) = min(min(corrcoef(simpleseismic,seismic_val)));
    end

    % 优选n口井.利用相关系数阈值
    tempabsCORcoef=abs(tempCORcoef);
    location=[];
    val=[];
    num = 0;%选出的井数量
    for k = 1:well_number2
        if tempabsCORcoef(1,k)>coefval
            location=[location;k];
            val=[val;tempabsCORcoef(1,k)];
            num = num+1;
        end
    end
    
    if num==0
        num = tempnum;
        temp_well_data = zeros(resample_point2,5,num);
        temp_seismic_data = zeros(resample_point2,num);
        val = zeros(num,1);
        location = zeros(num,1);
        for k = 1:num
            [val(k,1),location(k,1)] = max(tempabsCORcoef);
            temp_well_data(:,:,k) = well_data2(:,1:5,location(k,1));
            temp_seismic_data(:,k) = near_well_seismic_data2(:,location(k,1));
            tempabsCORcoef(1,location(k,1)) = 0;
        end
    else
        temp_well_data = zeros(resample_point2,5,num);
        temp_seismic_data = zeros(resample_point2,num);
        for k = 1:num
            temp_well_data(:,:,k) = well_data2(:,1:5,location(k,1));
            temp_seismic_data(:,k) = near_well_seismic_data2(:,location(k,1));
        end
    end
     
    if num>=2 %将相关系数的分布从新归置
        maxval = max(max(val));
        minval = min(min(val));
        max_minlen = maxval-minval;
        for kk = 1:num
            val(kk,1) = (val(kk,1)-minval)/max_minlen;
        end 
    else
        val(:)=1;
    end
   
    %利用相关系数插值
    valsum = sum(val);
    inter_impedence = zeros(wellpoint2,1);
    for k = 1:num
        val(k,1) = val(k,1)/valsum;
        inter_impedence = inter_impedence+val(k,1)*well_data2(:,DPS,location(k,1));
    end
    temp_Impedence_data2(:,i)=inter_impedence;
    
end
%% 写成segy数据
temp_Impedence_data12=[temp_Impedence_data1;temp_Impedence_data2];
temp_Impedence_data3=reshape(temp_Impedence_data12,[200,631,501]);

temp_Impedence_data22=zeros(size(temp_Impedence_data3));
Impedence_data=zeros(size(seismic_data2));
%%%%%%%%%%%%%导入地震数据信息%%%%%%%%%%%%%%
[seismic,bic_header,binary_header]=read_segy_file('./20200310/cdp_stack_3600_4000newT1_T31ms.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %读入地震数据

if isfilter 
    Wn=0.1;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % 纵向滤波  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % 纵向滤波  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[200,631*501]);
for i=1:trace_number
    up=1;
    down=1;
    for j=1:point_number2
        if seismic_data1(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up+10:point_number2
        if seismic_data1(k,i)==0
            down=k-1;
            break;
        end 
    end
    for k=down+10:point_number2
        if seismic_data2(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint2+wellpoint1);
    useful_len_seismic = linspace(1,useful_len,useful_len);
    %将结果插值成原来的点数
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %依次循环完每一道
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0328T1_T31ms_one_mc0.8_well10_filter0.1_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

if isfilter 
    Wn=0.05;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % 纵向滤波  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % 纵向滤波  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[200,631*501]);
for i=1:trace_number
    up=1;
    down=1;
    for j=1:point_number2
        if seismic_data1(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up+10:point_number2
        if seismic_data1(k,i)==0
            down=k-1;
            break;
        end 
    end
    for k=down+10:point_number2
        if seismic_data2(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint2+wellpoint1);
    useful_len_seismic = linspace(1,useful_len,useful_len);
    %将结果插值成原来的点数
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %依次循环完每一道
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0328T1_T31ms_one_mc0.8_well10_filter0.05_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

if isfilter 
    Wn=0.2;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % 纵向滤波  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % 纵向滤波  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[200,631*501]);
for i=1:trace_number
    up=1;
    down=1;
    for j=1:point_number2
        if seismic_data1(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up+10:point_number2
        if seismic_data1(k,i)==0
            down=k-1;
            break;
        end 
    end
    for k=down+10:point_number2
        if seismic_data2(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint2+wellpoint1);
    useful_len_seismic = linspace(1,useful_len,useful_len);
    %将结果插值成原来的点数
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %依次循环完每一道
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0328T1_T31ms_one_mc0.8_well10_filter0.2_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

if isfilter 
    Wn=0.4;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % 纵向滤波  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % 纵向滤波  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[200,631*501]);
for i=1:trace_number
    up=1;
    down=1;
    for j=1:point_number2
        if seismic_data1(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up+10:point_number2
        if seismic_data1(k,i)==0
            down=k-1;
            break;
        end 
    end
    for k=down+10:point_number2
        if seismic_data2(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len = down-up+1;
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint2+wellpoint1);
    useful_len_seismic = linspace(1,useful_len,useful_len);
    %将结果插值成原来的点数
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %依次循环完每一道
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0328T1_T31ms_one_mc0.8_well10_filter0.4_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

for kk=1
    temp_Impedence_data11=reshape(temp_Impedence_data1,[100,631,501]);
    temp_Impedence_data22=reshape(temp_Impedence_data2,[100,631,501]);
    temp_Impedence_data111=zeros(size(temp_Impedence_data11));
    temp_Impedence_data222=zeros(size(temp_Impedence_data22));
    Impedence_data=zeros(size(seismic_data2));
    %%%%%%%%%%%%%导入地震数据信息%%%%%%%%%%%%%%
    [seismic,bic_header,binary_header]=read_segy_file('./20200310/cdp_stack_3600_4000newT1_T31ms.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %读入地震数据

    if isfilter 
        Wn=0.1;
        for i=1:501
            temp_Impedence_data222(:,:,i) = csFilterProfile(temp_Impedence_data22(:,:,i), Wn, 'h');   % 纵向滤波  
            temp_Impedence_data111(:,:,i) = csFilterProfile(temp_Impedence_data11(:,:,i), Wn, 'h');   % 纵向滤波  
        end
        for i=1:631
            temp_Impedence_data222(:,i,:) = csFilterProfile(temp_Impedence_data222(:,i,:), Wn, 'h');   % 纵向滤波  
            temp_Impedence_data111(:,i,:) = csFilterProfile(temp_Impedence_data111(:,i,:), Wn, 'h');   % 纵向滤波  
        end
    end
    temp_Impedence_data2222=reshape(temp_Impedence_data222,[100,631*501]);
    temp_Impedence_data1111=reshape(temp_Impedence_data111,[100,631*501]);
    for i=1:trace_number
        up=1;
        down=1;
        for j=1:point_number2
            if seismic_data1(j,i)~=0
                up=j;
                break;
            end 
        end
        for k=up+10:point_number2
            if seismic_data1(k,i)==0
                down=k-1;
                break;
            end 
        end
        useful_len = down-up+1;
        inter_useful_len_seismic = linspace(1,useful_len,wellpoint1);
        useful_len_seismic = linspace(1,useful_len,useful_len);
        %将结果插值成原来的点数
        inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data1111(:,i)',useful_len_seismic,'spline');
        inter_impedence2 = inter_impedence2';
        Impedence_data(up:down,i) = inter_impedence2';
        
        up=1;
        down=1;
        for j=1:point_number2
            if seismic_data2(j,i)~=0
                up=j;
                break;
            end 
        end
        for k=up+10:point_number2
            if seismic_data2(k,i)==0
                down=k-1;
                break;
            end 
        end
        useful_len = down-up+1;
        inter_useful_len_seismic = linspace(1,useful_len,wellpoint2);
        useful_len_seismic = linspace(1,useful_len,useful_len);
        %将结果插值成原来的点数
        inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2222(:,i)',useful_len_seismic,'spline');
        inter_impedence2 = inter_impedence2';
        Impedence_data(up:down,i) = inter_impedence2';
        %依次循环完每一道
    end
    seismic.traces=Impedence_data;
    write_segy_file(seismic,'./LNresult/0328T1_T31ms_one_mc0.8_well10_filter0.1_Vp_Vs_2.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

    if isfilter 
        Wn=0.05;
        for i=1:501
            temp_Impedence_data222(:,:,i) = csFilterProfile(temp_Impedence_data22(:,:,i), Wn, 'h');   % 纵向滤波  
            temp_Impedence_data111(:,:,i) = csFilterProfile(temp_Impedence_data11(:,:,i), Wn, 'h');   % 纵向滤波  
        end
        for i=1:631
            temp_Impedence_data222(:,i,:) = csFilterProfile(temp_Impedence_data222(:,i,:), Wn, 'h');   % 纵向滤波  
            temp_Impedence_data111(:,i,:) = csFilterProfile(temp_Impedence_data111(:,i,:), Wn, 'h');   % 纵向滤波  
        end
    end
    temp_Impedence_data2222=reshape(temp_Impedence_data222,[100,631*501]);
    temp_Impedence_data1111=reshape(temp_Impedence_data111,[100,631*501]);
    for i=1:trace_number
        up=1;
        down=1;
        for j=1:point_number2
            if seismic_data1(j,i)~=0
                up=j;
                break;
            end 
        end
        for k=up+10:point_number2
            if seismic_data1(k,i)==0
                down=k-1;
                break;
            end 
        end
        useful_len = down-up+1;
        inter_useful_len_seismic = linspace(1,useful_len,wellpoint1);
        useful_len_seismic = linspace(1,useful_len,useful_len);
        %将结果插值成原来的点数
        inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data1111(:,i)',useful_len_seismic,'spline');
        inter_impedence2 = inter_impedence2';
        Impedence_data(up:down,i) = inter_impedence2';
        
        up=1;
        down=1;
        for j=1:point_number2
            if seismic_data2(j,i)~=0
                up=j;
                break;
            end 
        end
        for k=up+10:point_number2
            if seismic_data2(k,i)==0
                down=k-1;
                break;
            end 
        end
        useful_len = down-up+1;
        inter_useful_len_seismic = linspace(1,useful_len,wellpoint2);
        useful_len_seismic = linspace(1,useful_len,useful_len);
        %将结果插值成原来的点数
        inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2222(:,i)',useful_len_seismic,'spline');
        inter_impedence2 = inter_impedence2';
        Impedence_data(up:down,i) = inter_impedence2';
        %依次循环完每一道
    end
    seismic.traces=Impedence_data;
    write_segy_file(seismic,'./LNresult/0328T1_T31ms_one_mc0.8_well10_filter0.05_Vp_Vs_2.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

    if isfilter 
        Wn=0.2;
        for i=1:501
            temp_Impedence_data222(:,:,i) = csFilterProfile(temp_Impedence_data22(:,:,i), Wn, 'h');   % 纵向滤波  
            temp_Impedence_data111(:,:,i) = csFilterProfile(temp_Impedence_data11(:,:,i), Wn, 'h');   % 纵向滤波  
        end
        for i=1:631
            temp_Impedence_data222(:,i,:) = csFilterProfile(temp_Impedence_data222(:,i,:), Wn, 'h');   % 纵向滤波  
            temp_Impedence_data111(:,i,:) = csFilterProfile(temp_Impedence_data111(:,i,:), Wn, 'h');   % 纵向滤波  
        end
    end
    temp_Impedence_data2222=reshape(temp_Impedence_data222,[100,631*501]);
    temp_Impedence_data1111=reshape(temp_Impedence_data111,[100,631*501]);
    for i=1:trace_number
        up=1;
        down=1;
        for j=1:point_number2
            if seismic_data1(j,i)~=0
                up=j;
                break;
            end 
        end
        for k=up+10:point_number2
            if seismic_data1(k,i)==0
                down=k-1;
                break;
            end 
        end
        useful_len = down-up+1;
        inter_useful_len_seismic = linspace(1,useful_len,wellpoint1);
        useful_len_seismic = linspace(1,useful_len,useful_len);
        %将结果插值成原来的点数
        inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data1111(:,i)',useful_len_seismic,'spline');
        inter_impedence2 = inter_impedence2';
        Impedence_data(up:down,i) = inter_impedence2';
        
        up=1;
        down=1;
        for j=1:point_number2
            if seismic_data2(j,i)~=0
                up=j;
                break;
            end 
        end
        for k=up+10:point_number2
            if seismic_data2(k,i)==0
                down=k-1;
                break;
            end 
        end
        useful_len = down-up+1;
        inter_useful_len_seismic = linspace(1,useful_len,wellpoint2);
        useful_len_seismic = linspace(1,useful_len,useful_len);
        %将结果插值成原来的点数
        inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2222(:,i)',useful_len_seismic,'spline');
        inter_impedence2 = inter_impedence2';
        Impedence_data(up:down,i) = inter_impedence2';
        %依次循环完每一道
    end
    seismic.traces=Impedence_data;
    write_segy_file(seismic,'./LNresult/0328T1_T31ms_one_mc0.8_well10_filter0.2_Vp_Vs_2.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

    if isfilter 
        Wn=0.4;
        for i=1:501
            temp_Impedence_data222(:,:,i) = csFilterProfile(temp_Impedence_data22(:,:,i), Wn, 'h');   % 纵向滤波  
            temp_Impedence_data111(:,:,i) = csFilterProfile(temp_Impedence_data11(:,:,i), Wn, 'h');   % 纵向滤波  
        end
        for i=1:631
            temp_Impedence_data222(:,i,:) = csFilterProfile(temp_Impedence_data222(:,i,:), Wn, 'h');   % 纵向滤波  
            temp_Impedence_data111(:,i,:) = csFilterProfile(temp_Impedence_data111(:,i,:), Wn, 'h');   % 纵向滤波  
        end
    end
    temp_Impedence_data2222=reshape(temp_Impedence_data222,[100,631*501]);
    temp_Impedence_data1111=reshape(temp_Impedence_data111,[100,631*501]);
    for i=1:trace_number
        up=1;
        down=1;
        for j=1:point_number2
            if seismic_data1(j,i)~=0
                up=j;
                break;
            end 
        end
        for k=up+10:point_number2
            if seismic_data1(k,i)==0
                down=k-1;
                break;
            end 
        end
        useful_len = down-up+1;
        inter_useful_len_seismic = linspace(1,useful_len,wellpoint1);
        useful_len_seismic = linspace(1,useful_len,useful_len);
        %将结果插值成原来的点数
        inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data1111(:,i)',useful_len_seismic,'spline');
        inter_impedence2 = inter_impedence2';
        Impedence_data(up:down,i) = inter_impedence2';
        
        up=1;
        down=1;
        for j=1:point_number2
            if seismic_data2(j,i)~=0
                up=j;
                break;
            end 
        end
        for k=up+10:point_number2
            if seismic_data2(k,i)==0
                down=k-1;
                break;
            end 
        end
        useful_len = down-up+1;
        inter_useful_len_seismic = linspace(1,useful_len,wellpoint2);
        useful_len_seismic = linspace(1,useful_len,useful_len);
        %将结果插值成原来的点数
        inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2222(:,i)',useful_len_seismic,'spline');
        inter_impedence2 = inter_impedence2';
        Impedence_data(up:down,i) = inter_impedence2';
        %依次循环完每一道
    end
    seismic.traces=Impedence_data;
    write_segy_file(seismic,'./LNresult/0328T1_T31ms_one_mc0.8_well10_filter0.4_Vp_Vs_2.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

    
    
end

%% 将地震数据的频带加到插值好的波阻抗频带