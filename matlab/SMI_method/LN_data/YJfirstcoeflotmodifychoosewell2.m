%% ʵ��SMI�Ľ�����ȡʱ�������ϵ���ֵ
%֮ǰѡ�������ϵ��ߵ�ǰ15�ھ�������ѡ���ķ�ʽΪ�뾶��Χ�ںͿ����ϵ�����ֵ��
% 2020/3/14
% ��ͦ
clc;clear;
coefval = 0.6; %���ϵ�����?
T_W = 10; % ʱ����С
resample_point1 = 100; % �ز������?
number_T_W1 = resample_point1/T_W; %ʱ������
resample_point2 = 100; % �ز������?
number_T_W2 = resample_point2/T_W; %ʱ������
tempnum = 9; % ���һ�ھ���ûѡ����������ǰnum�ھ�����ֵ
DPS = 4;%Den5��P2��S3��G1��VP/Vs4,seismic6
isfilter=1;
% h=waitbar(0,'please wait');
CoreNum=20; %the number of cpu 5
if matlabpool('size')<=0  
    matlabpool(CoreNum);
else  
    disp('matlab pool already started');
end
%% ��ȡ����ݺ͵������ �ߵ��� ��λ
for k=1
    % ����
    seismic_data1=load('cdp_stack1500offset1msT1_T2smooth.mat');    
    seismic_data1=seismic_data1.cdp_stack1500offset1msT1_T2smooth;
    point_number1 = size(seismic_data1,1);
    trace_number = size(seismic_data1,2);
    % �����?
    load('welldataT1_T2nomal.mat');%���ز⾮ֵ
    well_data1=welldataT1_T2nomal;
    wellpoint1 = size(well_data1,1); 
    near_well_seismic_data1 = well_data1(:,6,:);
    well_number1 = size(well_data1,3);
    % �Ծ���ݽ����˲�?
    if 1
        Wn=0.4;
        for i=1:well_number1
            well_data1(:,DPS,i)= csFilterProfile(well_data1(:,DPS,i), Wn, 'v');   % �����˲�  
        end 
    end

    
    
end
%% ��ֵ���?

temp_Impedence_data1=zeros(wellpoint1,trace_number);

parfor i=1:trace_number
    %���ӽ��?
%     str=['������...',num2str(i/trace_number*100),'%'];
%     waitbar(i/trace_number,h,str)
        i
    % ��ȡһ��������ݵ���Ч��ݶ�  seismic_val
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
    % ��ֵ����������ݲ�ֵ�ɾ����һ��ĵ���?
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint1);
    seismic_val=interp1(useful_len_seismic,seismic_val,inter_useful_len_seismic,'spline');


    %% ��ѡn�ھ�������ֻ����n�ھ�
    % ��ȡһ��������ݵ���Ч��ݶ�  seismic_val�����о���������ϵ��
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
    
    % ��ѡn�ھ�.�������ϵ�����?
    tempabsCORcoef=abs(tempCORcoef);
    location2=[];
    num = 0;%ѡ���ľ�����
    for k = 1:well_number1
        if tempabsCORcoef(1,k)>coefval
            location2=[location2;k];
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
            temp_well_data(:,:,k) = well_data1(:,1:5,location2(k,1));
            temp_seismic_data(:,k) = near_well_seismic_data1(:,location2(k,1));
        end
    end

    %% ȡʱ�������ϵ��?��һ��ʱ�������ϵ����Ϊʱ���м������ϵ��
    
    traceCORcoef = zeros(number_T_W1,num);
    for k2=1:resample_point1/T_W
        %��ʱ��
        Time_Window = (k2-1)*T_W+1:k2*T_W;
%         Center_point = ceil((k-1)*T_W+T_W/2);
        
        % ��ȡ������ݵ���Ч��ݶ�  seismic_val
        tempseismic_val=seismic_val(Time_Window,1);
        % ����ÿ�ھ��������ݵ����ϵ��?��Ҫ����һ���������?
        tempseismic_val = mapminmax(tempseismic_val');
        tempseismic_val = tempseismic_val';
        for K2 = 1:num
            simplewell_seismic_val = near_well_seismic_data1(Time_Window,K2);
            simplewell_seismic_val = simplewell_seismic_val';
            simplewell_seismic_val = mapminmax(simplewell_seismic_val);
            simplewell_seismic_val = simplewell_seismic_val';
            traceCORcoef(k2,K2) = min(min(corrcoef(simplewell_seismic_val,tempseismic_val)));
        end
    end
    
    %%  �����ϵ���ֵ����ɺ�һ�����һ��ĵ���?
    coef = zeros(resample_point1,num);
    sampletraceCORcoef = zeros(number_T_W1+2,1);
    inter_coef = zeros(number_T_W1+2,1);
    for K3 = 1:num
        sampletraceCORcoef(2:number_T_W1+1,1) = abs(traceCORcoef(:,K3));
        sampletraceCORcoef(1,1) = sampletraceCORcoef(2,1);
        sampletraceCORcoef(number_T_W1+2,1) = sampletraceCORcoef(number_T_W1+1,1);
        
        inter_coef(2:number_T_W1+1,1) = linspace(ceil(T_W/2),ceil((number_T_W1-1)*T_W+T_W/2),number_T_W1);
        inter_coef(1,1) = 1;
        inter_coef(number_T_W1+2,1) = resample_point1;
        inter_coef2 = linspace(1,resample_point1,resample_point1);
        
        coef(:,K3)=interp1(inter_coef,sampletraceCORcoef,inter_coef2,'spline');
        % �����ϵ����в�ֵ����ֵ��һ��������ϵ��
        maxcoef = max(max(sampletraceCORcoef));
%         for tempnumber = 1:resample_point
%             if coef(tempnumber,K1)>maxcoef
%                 coef(tempnumber,K1) = maxcoef;
%             end
%         end
        %%��Ϊ���ϵ���и���ͳ���1�������Դ���һ��   
    end
    
    %% ����ֵ
    inter_impedence = zeros(resample_point1,1);

    for sample_point = 1:resample_point1
        if num>=2  %�����ϵ��ķֲ����¹���
            maxval = max(max(coef(sample_point,:)));
            minval = min(min(coef(sample_point,:)));
            max_minlen = maxval-minval;
            for kk = 1:num
                coef(sample_point,kk) = (coef(sample_point,kk)-minval)/max_minlen;
            end 
        else
            coef(:,:)=1;
        end
        
        valsum = sum(coef(sample_point,:));
        for k_well2 = 1:num
            coef(sample_point,k_well2) = coef(sample_point,k_well2)/valsum;
            inter_impedence(sample_point,1) = inter_impedence(sample_point,1)+coef(sample_point,k_well2)*temp_well_data(sample_point,DPS,k_well2);
        end
    end
    %������ֵ��ԭ���ĵ���
    temp_Impedence_data1(:,i)=inter_impedence;
    
end
    
for k=1
    % ����
    seismic_data2=load('cdp_stack1500offset1msT2_T3_smooth.mat');    
    seismic_data2=seismic_data2.cdp_stack1500offset1msT2_T3_smooth;
    point_number2 = size(seismic_data2,1);
    trace_number = size(seismic_data2,2);
    % �����?
    load('welldataT2_T3nomal.mat');%���ز⾮ֵ
    well_data2=welldataT2_T3nomal;
    wellpoint2 = size(well_data2,1); 
    near_well_seismic_data2 = well_data2(:,6,:);
    well_number2 = size(well_data2,3);
    % �Ծ���ݽ����˲�?
    if 1
        Wn=0.4;
        for i=1:well_number2
            well_data2(:,DPS,i)= csFilterProfile(well_data2(:,DPS,i), Wn, 'v');   % �����˲�  
        end 
    end

    
    
end
%% ��ֵ���?

temp_Impedence_data2=zeros(wellpoint2,trace_number);

parfor i=1:trace_number
    %���ӽ��?
%     str=['������...',num2str(i/trace_number*100),'%'];
%     waitbar(i/trace_number,h,str)
        i
    % ��ȡһ��������ݵ���Ч��ݶ�  seismic_val
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
    % ��ֵ����������ݲ�ֵ�ɾ����һ��ĵ���?
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint2);
    seismic_val=interp1(useful_len_seismic,seismic_val,inter_useful_len_seismic,'spline');


    %% ��ѡn�ھ�������ֻ����n�ھ�
    % ��ȡһ��������ݵ���Ч��ݶ�  seismic_val�����о���������ϵ��
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
    
    % ��ѡn�ھ�.�������ϵ�����?
    tempabsCORcoef=abs(tempCORcoef);
    location2=[];
    num = 0;%ѡ���ľ�����
    for k = 1:well_number2
        if tempabsCORcoef(1,k)>coefval
            location2=[location2;k];
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
            temp_well_data(:,:,k) = well_data2(:,1:5,location2(k,1));
            temp_seismic_data(:,k) = near_well_seismic_data2(:,location2(k,1));
        end
    end

    %% ȡʱ�������ϵ��?��һ��ʱ�������ϵ����Ϊʱ���м������ϵ��
    
    traceCORcoef = zeros(number_T_W2,num);
    for k2=1:resample_point2/T_W
        %��ʱ��
        Time_Window = (k2-1)*T_W+1:k2*T_W;
%         Center_point = ceil((k-1)*T_W+T_W/2);
        
        % ��ȡ������ݵ���Ч��ݶ�  seismic_val
        tempseismic_val=seismic_val(Time_Window,1);
        % ����ÿ�ھ��������ݵ����ϵ��?��Ҫ����һ���������?
        tempseismic_val = mapminmax(tempseismic_val');
        tempseismic_val = tempseismic_val';
        for K2 = 1:num
            simplewell_seismic_val = near_well_seismic_data2(Time_Window,K2);
            simplewell_seismic_val = simplewell_seismic_val';
            simplewell_seismic_val = mapminmax(simplewell_seismic_val);
            simplewell_seismic_val = simplewell_seismic_val';
            traceCORcoef(k2,K2) = min(min(corrcoef(simplewell_seismic_val,tempseismic_val)));
        end
    end
    
    %%  �����ϵ���ֵ����ɺ�һ�����һ��ĵ���?
    coef = zeros(resample_point2,num);
    sampletraceCORcoef = zeros(number_T_W2+2,1);
    inter_coef = zeros(number_T_W2+2,1);
    for K3 = 1:num
        sampletraceCORcoef(2:number_T_W2+1,1) = abs(traceCORcoef(:,K3));
        sampletraceCORcoef(1,1) = sampletraceCORcoef(2,1);
        sampletraceCORcoef(number_T_W2+2,1) = sampletraceCORcoef(number_T_W2+1,1);
        
        inter_coef(2:number_T_W2+1,1) = linspace(ceil(T_W/2),ceil((number_T_W2-1)*T_W+T_W/2),number_T_W2);
        inter_coef(1,1) = 1;
        inter_coef(number_T_W2+2,1) = resample_point2;
        inter_coef2 = linspace(1,resample_point2,resample_point2);
        
        coef(:,K3)=interp1(inter_coef,sampletraceCORcoef,inter_coef2,'spline');
        % �����ϵ����в�ֵ����ֵ��һ��������ϵ��
        maxcoef = max(max(sampletraceCORcoef));
%         for tempnumber = 1:resample_point
%             if coef(tempnumber,K1)>maxcoef
%                 coef(tempnumber,K1) = maxcoef;
%             end
%         end
        %%��Ϊ���ϵ���и���ͳ���1�������Դ���һ��   
    end
    
    %% ����ֵ
    inter_impedence = zeros(resample_point2,1);

    for sample_point = 1:resample_point2
        if num>=2  %�����ϵ��ķֲ����¹���
            maxval = max(max(coef(sample_point,:)));
            minval = min(min(coef(sample_point,:)));
            max_minlen = maxval-minval;
            for kk = 1:num
                coef(sample_point,kk) = (coef(sample_point,kk)-minval)/max_minlen;
            end 
        else
            coef(:,:)=1;
        end
        
        valsum = sum(coef(sample_point,:));
        for k_well2 = 1:num
            coef(sample_point,k_well2) = coef(sample_point,k_well2)/valsum;
            inter_impedence(sample_point,1) = inter_impedence(sample_point,1)+coef(sample_point,k_well2)*temp_well_data(sample_point,DPS,k_well2);
        end
    end
    %������ֵ��ԭ���ĵ���
    temp_Impedence_data2(:,i)=inter_impedence;
    
end

%% д��segy���?
temp_Impedence_data12=[temp_Impedence_data1;temp_Impedence_data2];
temp_Impedence_data3=reshape(temp_Impedence_data12,[150,631,501]);

temp_Impedence_data22=zeros(size(temp_Impedence_data3));
Impedence_data=zeros(size(seismic_data2));
%%%%%%%%%%%%%�������������Ϣ%%%%%%%%%%%%%%
%%%%%%%%%%%%%������������Ϣ%%%%%%%%%%%%%%
[seismic,bic_header,binary_header]=read_segy_file('cdp_stack_20200310_1msT1_T3.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %����������

if isfilter 
    Wn=0.1;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % �����˲�  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % �����˲�  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[150,631*501]);
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
    %�������ֵ��ԭ���ĵ���
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %����ѭ����ÿһ��
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0326T1_T31ms_lot_mc0.6_well8_filter0.1_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������

if isfilter 
    Wn=0.05;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % �����˲�  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % �����˲�  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[150,631*501]);
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
    %�������ֵ��ԭ���ĵ���
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %����ѭ����ÿһ��
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0326T1_T31ms_lot_mc0.6_well8_filter0.05_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������

if isfilter 
    Wn=0.2;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % �����˲�  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % �����˲�  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[150,631*501]);
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
    %�������ֵ��ԭ���ĵ���
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %����ѭ����ÿһ��
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0326T1_T31ms_lot_mc0.6_well8_filter0.2_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������

if isfilter 
    Wn=0.4;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % �����˲�  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % �����˲�  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[150,631*501]);
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
    %�������ֵ��ԭ���ĵ���
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %����ѭ����ÿһ��
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0326T1_T31ms_lot_mc0.6_well8_filter0.4_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������

%% ��������ݵ�Ƶ��ӵ�,[200,631*501]��ֵ�õĲ��迹Ƶ��