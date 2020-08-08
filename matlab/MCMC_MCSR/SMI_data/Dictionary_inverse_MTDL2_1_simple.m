% clc;
% clear all;
%% load and define
%%%file
%%%parameter
% T0=20;   
% markovlen=20;
endline=0.1;

K1=39;
trace=68;



%% load data

waveletFreq = 30;
wave = s_create_wavelet({'type','ricker'}, {'frequencies',waveletFreq}, {'step', 1}, {'wlength',101});
wavelet = wave.traces;
% wavelet=load('.\SeismicData\wavelet1225.txt');                                       %%load wavelet
impedence=load('.\SeismicData\inline99_initalmodel_52.mat');                     %%load varance
seismic_data=load('.\SeismicData\inline99_seismic.txt');                         %%load seismic
variance=load('.\SeismicData\inline99_varance_52.mat');                          %%load inital impendence
temple_DIC=load('.\Dictionaries\DIC_50_1500_NWell_104_filt_1.00.mat');    %%load dictionaary

load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
well=zeros(2001,1);
for i=1:503
    if Well_timeData(i,K1)==fix(Well_timeData(i,K1))&&Well_timeData(i,K1)~=0
        well(fix(Well_timeData(i,K1)),1)=Well_valueData(i,K1);
    end
end
for sampling_number=1:2001      
    if( seismic_data(sampling_number,trace)~=0)
        initial_point=sampling_number;
        break;
    end
end
for sampling_number=sampling_number:2001   
    if( seismic_data(sampling_number,trace )~=0)
       termination_point = sampling_number;
    end
end
useful_len = termination_point-initial_point+1;
reflect=zeros(useful_len-1,1);           %%%
for l=1:useful_len-1                     %%%reflect
    reflect(l)=(well(initial_point+l,1)-well(initial_point+l-1,1))/(well(initial_point+l,1)+well(initial_point+l-1,1));
end
syn_seis=conv(wavelet,reflect);         %%%
lensyn_seis=length(syn_seis);           %%%length of syn_seis
syn_seis=syn_seis(round((lensyn_seis-useful_len)/2+2):round((lensyn_seis-useful_len)/2+useful_len+1));
% 添加噪声
noiseType = 1;
noiseDB = 4;
noiseOption = {'gs','lp'};
postNoise = wyjGenNormNoise2D(syn_seis, noiseDB, noiseType);
seismic_data(initial_point:termination_point,trace) = syn_seis;   
seismic_data=seismic_data(1:2001,trace);

initial_impedence=impedence.initial_impedence(1:2001,trace);
% Wn = 0.5;
% smooth=csFilterProfile(initial_impedence(initial_point:termination_point,1), Wn, 'v'); % 纵向滤波
% initial_impedence(initial_point:termination_point,1)=smooth;

inversion_variance=variance.inversion_variance(1:2001,trace);

DIC=temple_DIC.DIC;
%% Define the size of the matrix to store the results
[sampNum, traceNum]=size(initial_impedence);
saveBestX=zeros(sampNum, traceNum);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for K=1                      
    fprintf('trace：%d\n',K);
    %% 
    inim0_tmp=initial_impedence(:,K);     %%%inital impendence
    for sampling_number=1:length(inim0_tmp)      
        if( inim0_tmp(sampling_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:length(inim0_tmp)   
        if( inim0_tmp(sampling_number)~=0)
           termination_point=sampling_number;
        end
    end
    uesful_len=termination_point-initial_point+1;    %%%useful length
    inim0=zeros(uesful_len,1);                       %%%inital impendence 
    inivar=zeros(uesful_len,1);                      %%%varance
    d_seismic=zeros(uesful_len,1);                   %%%seismic data
    for f_number=1:uesful_len                        %%%useful length
        inim0(f_number)= inim0_tmp(f_number+initial_point-1);
        inivar(f_number)=inversion_variance(f_number+initial_point-1,K);
        d_seismic(f_number)=seismic_data(f_number+initial_point-1,K);
    end
    [inim0,~]=MT_DL2_1modify(inim0,inivar,wavelet,d_seismic,T0,markovlen,DIC,endline);
    result_tmp=zeros(length(inim0_tmp),1);
    for MM=1:uesful_len
        result_tmp(MM+initial_point-1)=inim0(MM);
    end
    saveBestX(:,K)=result_tmp;
 end
 %% output
dicSavePath = sprintf('./ResultsData_simple_richer/add_syn/MTDL_2well_39_trace_68/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52.mat', markovlen, T0);
save(dicSavePath, 'saveBestX');