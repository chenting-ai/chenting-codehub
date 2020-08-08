clc;
close all;
clear all;
%% 加载数据
load('.\SeismicData\seismic_data.mat');   %seismic_data
load('.\Load_WellData\Xline_Inline_number.mat');
temple_DIC=load('.\Dictionaries_alldata\DIC_50_1500_NWell_52_filt_1.00.mat');     %%load dictionaary
DIC=temple_DIC.DIC;
%%
Xline=Xline_Inline_number(1,:);
Inline=Xline_Inline_number(2,:);
seismic_temp=[];
for i=1:26
    seismic_temp=[seismic_temp,seismic_data(:,110*(Inline(1,4*i-3)-1)+Xline(1,4*i-3))]; 
end

load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
well_temp=zeros(2001,26);

for K1=1:26;
    for i=1:503
        if Well_timeData(i,4*K1-3)==fix(Well_timeData(i,4*K1-3))&&Well_timeData(i,4*K1-3)~=0
            well_temp(fix(Well_timeData(i,K1))+1,K1)=Well_valueData(i,4*K1-3);
        end
    end
end

well=zeros(62,26);
for j=1:26
    for sampling_number=1:2001      
        if( seismic_temp(sampling_number,j)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:2001  
        if( seismic_temp(sampling_number,j )~=0)
           termination_point=sampling_number;
        end
    end
    useful_len = termination_point-initial_point+1;
    well(1:useful_len,j)=well_temp(initial_point:termination_point,j);
end

mark_atoms=zeros(size(well,1),size(DIC,2));
Atomic_length = size(DIC,1);
T=10;
hwait=waitbar(0,'运行进度');
for well_number=1:26
    perstr=fix(well_number/26*100);
    str=['运行进度',num2str(perstr),'%'];
    waitbar(well_number/26,hwait,str);
    
    for sampling_number=1:size(well,1)      
        if( well(sampling_number,well_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:size(well,1)  
        if( well(sampling_number,well_number )~=0)
           termination_point=sampling_number;
        end
    end
    len=termination_point-initial_point+1;
    data=well(initial_point:termination_point,well_number);
    if size(data,1)<Atomic_length
        continue;
    end
    for j=1:len

        if j>len
           break;
        end

        if j>len-Atomic_length/2
           jj=len-Atomic_length/2;
        elseif j<Atomic_length/2
           jj=Atomic_length/2;
        else
           jj=j;
        end
        
        temp_data=data(jj-Atomic_length/2+1:jj+Atomic_length/2,1);
        Spare=omp(DIC,temp_data,DIC'*DIC,T); 
        mark_atoms(j,:) = mark_atoms(j,:)+Spare';
    end
end

for i=1:size(well,1)
    for j=1:size(DIC,2)
        
        if mark_atoms(i,j)~=0
            mark_atoms(i,j)=1;
        end
    end
end
         
dicSavePath = sprintf('./Dictionaries_alldata/mark_atoms_%d_%dNWell_52.mat', size(DIC,1), size(DIC,2));%%%记得改后面208
save(dicSavePath, 'mark_atoms');
a = sum(mark_atoms);
b = sum(mark_atoms');


