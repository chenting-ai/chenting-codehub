%% 处理轮南数据，将数据整成神经网络可以用的格式
% 将地震数据，井数据插值成相同长度，记录归一化参数和线道号
clear;
clc;


%% 处理地震数据
% 拼接地震数据，并插值成相同长度
if 1
[seismic1,bic_header,binary_header]=read_segy_file('./20200310/cdp_stack1500offset1msT1_T3.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %读入地震数据
[lable1,bic_header,binary_header]=read_segy_file('./LNresult/0328T1_T31ms_lot_mc0.8_well10_filter0.05_Vp_Vs_2.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %读入地震数据
seismic_data = seismic1.traces;
lable_data = lable1.traces;
point_number1 = size(seismic_data,1);
trace_number = size(seismic_data,2);
seismic = zeros(100, trace_number);
lable = zeros(100, trace_number);
temp_seismic = zeros(point_number1, 1);

for i=1:trace_number
    i
    up1=1;
    down1=1;
    for j=1:point_number1
        if seismic_data(j,i)~=0
            up1=j;
            break;
        end 
    end
    for k=up1:point_number1
        if seismic_data(k,i)==0
            down1=k-1;
            break;
        end 
    end
    seismic_val = seismic_data(up1:down1, i);
    lable_val = lable_data(up1:down1, i);
    useful_len = down1-up1+1;
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,100);
    seismic_val=interp1(useful_len_seismic,seismic_val,inter_useful_len_seismic,'spline');
    lable_val=interp1(useful_len_seismic,lable_val,inter_useful_len_seismic,'spline');
    if find(lable_val==0)
        a=1
        break;
    end
    seismic(:, i) = seismic_val;
    lable(:, i) = lable_val;
end

max_seismic = max(max(seismic));
min_seismic = min(min(seismic))-0.0001;
len_seismic = max_seismic-min_seismic;
max_lable = max(max(lable));
min_lable = min(min(lable))-0.0001;
len_lable = max_lable-min_lable;

seismic = (seismic-min_seismic)/len_seismic;
lable = (lable-min_lable)/len_lable;

end
%% 处理井数据
DPS = 4;%Den5，P2，S3，G1，VP/Vs4,seismic6
load('.\LNusefuldata\welldataT1_T2_2.mat');%加载测井值
well_data1 = welldataT1_T2;
load('.\LNusefuldata\welldataT2_T3_2.mat');%加载测井值
well_data2 = welldataT2_T3;

if 1
    Wn=0.4;
    for i=1:size(welldataT1_T2, 3)
        well_data1(:,DPS,i)= csFilterProfile(well_data1(:,DPS,i), Wn, 'v');   % 纵向滤波  
        well_data2(:,DPS,i)= csFilterProfile(well_data2(:,DPS,i), Wn, 'v');   % 纵向滤波  
    end 
end

well_vp_vs = zeros(100, size(welldataT1_T2, 3));
well_seismic = zeros(100, size(welldataT1_T2, 3));
for j=1:size(welldataT1_T2, 3)
    temp_vp_vs = well_data1(:,DPS,j);
    temp_vp_vs = [temp_vp_vs; well_data2(:,DPS,j)];
    near_well_seismic_data = well_data1(:,6,j);
    near_well_seismic_data = [near_well_seismic_data;well_data2(:,6,j)];
    
    useful_len = size(temp_vp_vs, 1);
    useful_len_seismic = linspace(1, useful_len, useful_len);
    inter_useful_len_seismic = linspace(1, useful_len, 100);
    temp_vp_vs=interp1(useful_len_seismic, temp_vp_vs, inter_useful_len_seismic,'spline');
    well_vp_vs(:,j) = temp_vp_vs;
    near_well_seismic_data=interp1(useful_len_seismic, near_well_seismic_data, inter_useful_len_seismic,'spline');
    well_seismic(:,j) = near_well_seismic_data;
end

max_well_vp_vs = max(max(well_vp_vs));
min_well_vp_vs = min(min(well_vp_vs))-0.0001;
len_well_vp_vs = max_well_vp_vs-min_well_vp_vs;
max_well_seismic = max(max(well_seismic));
min_well_seismic = min(min(well_seismic))-0.0001;
len_well_seismic = max_well_seismic-min_well_seismic;

well_vp_vs = (well_vp_vs-min_well_vp_vs)/len_well_vp_vs;
well_seismic = (well_seismic-min_well_seismic)/len_well_seismic;

%%