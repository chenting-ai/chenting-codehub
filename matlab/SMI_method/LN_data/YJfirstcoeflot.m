%% ����SMI������LN���ݣ�ȡʱ�������ϵ����ֵ
% 2020/3/10
% ��ͦ
clc;clear;
num = 7; % ������ֵ�ľ�����
DPS = 4;%Den5��P2��S3��G1��VP/Vs4,seismic6
T_W = 10; % ʱ����С
resample_point = 100; % �ز�������
number_T_W = resample_point/T_W; %ʱ������
isfilter=1;%�˲���
%% ��ȡ�����ݺ͵�������
for k=1
    % ����
    seismic_data=load('.\LNusefuldata\cdp_stack_3600_4000newT1_T31ms.mat');    
    seismic_data=seismic_data.cdp_stack_3600_4000newT1_T31ms;
    point_number = size(seismic_data,1);
    trace_number = size(seismic_data,2);
    % ������
    load('.\LNusefuldata\well_data2.mat');%���ز⾮ֵ
    well_data=well_data2;
    wellpoint = size(well_data,1); 
    near_well_seismic_data = well_data(:,6,:);
    well_number = size(well_data,3);
    % �Ծ����ݽ����˲�
    if 1
        Wn=0.3;
        for i=1:well_number
            well_data(:,DPS,i)= csFilterProfile(well_data(:,DPS,i), Wn, 'v');   % �����˲�  
        end 
    end

    
    
end
%% ��ֵ����
temp_Impedence_data=zeros(wellpoint,trace_number);
% h=waitbar(0,'please wait');
CoreNum=5; %the number of cpu 5
if matlabpool('size')<=0  
    matlabpool('open','local',CoreNum);
else  
    disp('matlab pool already started');
end
parfor i=1:trace_number
    i
    %���ӽ���
%     str=['������...',num2str(i/trace_number*100),'%'];
%     waitbar(i/trace_number,h,str)
    
    % ��ȡһ���������ݵ���Ч���ݶ�  seismic_val
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
    all_point = size(seismic_data,1);
    % ��ֵ�����������ݲ�ֵ�ɾ�����һ���ĵ��� 
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint);
    seismic_val=interp1(useful_len_seismic,seismic_val,inter_useful_len_seismic,'spline');
    %% ��ѡn�ھ�������ֻ����n�ھ�
    
    % ��ȡһ���������ݵ���Ч���ݶ�  seismic_val�����о����������ϵ��
    tempCORcoef = zeros(1,well_number);
    seismic_val = mapminmax(seismic_val);
    seismic_val = seismic_val';
    for K1 = 1:well_number
        simpleseismic = near_well_seismic_data(:,K1);
        simpleseismic = simpleseismic';
        simpleseismic = mapminmax(simpleseismic);
        simpleseismic = simpleseismic';
        tempCORcoef(1,K1) = min(min(corrcoef(simpleseismic,seismic_val)));
    end
    
    % ��ѡn�����ϵ���ϸߵľ�
    tempabsCORcoef=abs(tempCORcoef);
    temp_well_data = zeros(resample_point,5,num);
    temp_seismic_data = zeros(resample_point,num);
    val = zeros(num,1);
    location = zeros(num,1);
    for k = 1:num
        [val(k,1),location(k,1)] = max(tempabsCORcoef);
        temp_well_data(:,:,k) = well_data(:,1:5,location(k,1));
        temp_seismic_data(:,k) = near_well_seismic_data(:,location(k,1));
        tempabsCORcoef(1,location(k,1)) = 0;
    end

    %% ȡʱ�������ϵ��,��һ��ʱ�������ϵ����Ϊʱ���м������ϵ��
    
    traceCORcoef = zeros(number_T_W,num);
    for k2=1:resample_point/T_W
        %��ʱ��
        Time_Window = (k2-1)*T_W+1:k2*T_W;
%         Center_point = ceil((k-1)*T_W+T_W/2);
        
        % ��ȡ�������ݵ���Ч���ݶ�  seismic_val
        tempseismic_val=seismic_val(Time_Window,1);
        % ����ÿ�ھ���������ݵ����ϵ�� ��Ҫ����һ���������
        tempseismic_val = mapminmax(tempseismic_val');
        tempseismic_val = tempseismic_val';
        for K2 = 1:num
            simplewell_seismic_val = near_well_seismic_data(Time_Window,K2);
            simplewell_seismic_val = simplewell_seismic_val';
            simplewell_seismic_val = mapminmax(simplewell_seismic_val);
            simplewell_seismic_val = simplewell_seismic_val';
            traceCORcoef(k2,K2) = min(min(corrcoef(simplewell_seismic_val,tempseismic_val)));
        end
    end
    
    %%  �����ϵ����ֵ����ɺ�һ������һ���ĵ���
    coef = zeros(resample_point,num);
    sampletraceCORcoef = zeros(number_T_W+2,1);
    inter_coef = zeros(number_T_W+2,1);
    for K3 = 1:num
        sampletraceCORcoef(2:number_T_W+1,1) = abs(traceCORcoef(:,K3));
        sampletraceCORcoef(1,1) = sampletraceCORcoef(2,1);
        sampletraceCORcoef(number_T_W+2,1) = sampletraceCORcoef(number_T_W+1,1);
        
        inter_coef(2:number_T_W+1,1) = linspace(ceil(T_W/2),ceil((number_T_W-1)*T_W+T_W/2),number_T_W);
        inter_coef(1,1) = 1;
        inter_coef(number_T_W+2,1) = resample_point;
        inter_coef2 = linspace(1,resample_point,resample_point);
        
        coef(:,K3)=interp1(inter_coef,sampletraceCORcoef,inter_coef2,'spline');
        % �����ϵ�����в�ֵ����ֵ��һ���������ϵ��
        maxcoef = max(max(sampletraceCORcoef));
%         for tempnumber = 1:resample_point
%             if coef(tempnumber,K1)>maxcoef
%                 coef(tempnumber,K1) = maxcoef;
%             end
%         end
        %%��Ϊ���ϵ���и����ͳ���1���������Դ���һ��   
    end
    
    %% ����ֵ
    inter_impedence = zeros(resample_point,1);

    for sample_point = 1:resample_point
        if 1  %�����ϵ���ķֲ����¹���
            maxval = max(max(coef(sample_point,:)));
            minval = min(min(coef(sample_point,:)));
            max_minlen = maxval-minval;
            for kk = 1:num
                coef(sample_point,kk) = (coef(sample_point,kk)-minval)/max_minlen;
            end 
        end
        
        valsum = sum(coef(sample_point,:));
        for k_well2 = 1:num
            coef(sample_point,k_well2) = coef(sample_point,k_well2)/valsum;
            inter_impedence(sample_point,1) = inter_impedence(sample_point,1)+coef(sample_point,k_well2)*temp_well_data(sample_point,DPS,k_well2);
        end
    end
    %�������ֵ��ԭ���ĵ���
    temp_Impedence_data(:,i)=inter_impedence;
    
end

%% д��segy����
temp_Impedence_data2=zeros(size(temp_Impedence_data));
Impedence_data=zeros(size(seismic_data));
%%%%%%%%%%%%%�������������Ϣ%%%%%%%%%%%%%%
[seismic,bic_header,binary_header]=read_segy_file('./20200310/cdp_stack_20200310_1msT1_T3.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %�����������


if isfilter 
    Wn=0.2;
    temp_Impedence_data2 = csFilterProfile(temp_Impedence_data, Wn, 'h');   % �����˲�  
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
    %�������ֵ��ԭ���ĵ���
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %����ѭ����ÿһ��
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0313T1_T31ms_lot_well6_filter0.2_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������

if isfilter 
    Wn=0.3;
    temp_Impedence_data2 = csFilterProfile(temp_Impedence_data, Wn, 'h');   % �����˲�  
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
    %�������ֵ��ԭ���ĵ���
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %����ѭ����ÿһ��
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0313T1_T31ms_lot_well6_filter0.3_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������

if isfilter 
    Wn=0.4;
    temp_Impedence_data2 = csFilterProfile(temp_Impedence_data, Wn, 'h');   % �����˲�  
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
    %�������ֵ��ԭ���ĵ���
    inter_impedence2=interp1(inter_useful_len_seismic,temp_Impedence_data2(:,i)',useful_len_seismic,'spline');
    inter_impedence2 = inter_impedence2';
    Impedence_data(up:down,i) = inter_impedence2';
    %����ѭ����ÿһ��
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./LNresult/0313T1_T31ms_lot_well6_filter0.4_Vp_Vs.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������






%% ���������ݵ�Ƶ���ӵ���ֵ�õĲ��迹Ƶ��