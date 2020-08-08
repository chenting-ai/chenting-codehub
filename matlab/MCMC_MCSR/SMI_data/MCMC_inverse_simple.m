clc;
clear all;

%% ��ȡ�ļ����������
%%%�ļ��У���ʼ���迹ģ�ͣ��������ݣ��������ݣ��Ӳ�
%%%�����У�
T0=20;    %%%��ʼ�¶�
markovlen=20;
endline=0.1;
%��ʱд��  ��ȡ�ļ�д�ñȽϼ�
K1=39;
trace=68;
%% ��ʼ���������ݺͳ�ʼ���迹
impedence=load('.\SeismicData\inline99_initalmodel_52.mat');    %%load varance
seismic_data=load('.\SeismicData\inline99_seismic.txt');        %%load seismic


waveletFreq = 30;
wave = s_create_wavelet({'type','ricker'}, {'frequencies',waveletFreq}, {'step', 1}, {'wlength',101});
wavelet = wave.traces;
% wavelet=load('.\SeismicData\wavelet1225.txt');                      %%load wavelet
variance=load('.\SeismicData\inline99_varance_52.mat');         %%load inital impendence

%%��99��inline 68Xline   �ǵ�39�ھ�
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%���ز⾮ֵ
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%���ز⾮ʱ�����
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
% �������
noiseType = 1;
noiseDB = 4;
noiseOption = {'gs','lp'};
postNoise = wyjGenNormNoise2D(syn_seis, noiseDB, noiseType);
seismic_data(initial_point:termination_point,trace) = syn_seis;   
seismic_data=seismic_data(1:2001,trace);


initial_impedence=impedence.initial_impedence(1:2001,trace);
% Wn = 0.5;
% smooth=csFilterProfile(initial_impedence(initial_point:termination_point,1), Wn, 'v'); % �����˲�
% initial_impedence(initial_point:termination_point,1)=smooth;


inversion_variance=variance.inversion_variance(1:2001,trace);


seismic_data=seismic_data';
initial_impedence=initial_impedence';
inversion_variance=inversion_variance';

%�����Ž���ľ����С
[row, col]=size(initial_impedence);
saveBestX=zeros(row,col);
% ERRORS=zeros(row,:);
%% ��������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for K=1                          %%%�������
    fprintf('������%d��\n',K);
    inim0_tmp=initial_impedence(K,:);  %%%��ʼ���迹
    inivar_tmp=inversion_variance(K,:);%%%����
    d_tmp=seismic_data(K,:);           %%%��������
    
    for sampling_number=1:length(inim0_tmp)          %%%�ӷ���ĵ㿪ʼ
        if(initial_impedence(K,sampling_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:length(inim0_tmp)          %%%�����һ���������
        if(initial_impedence(K,sampling_number)~=0)
           termination_point=sampling_number;
        end
    end
    uesful_len=termination_point-initial_point+1;              %%%��Ч���ݳ���
    inim0=zeros(uesful_len,1);               %%%���迹
    inivar=zeros(uesful_len,1);              %%%����
    d=zeros(uesful_len,1);
    bestX=zeros(uesful_len,1);               %%%���ݽ��
    for f_number=1:uesful_len
        inim0(f_number)=initial_impedence(K,f_number+initial_point-1);
        inivar(f_number)=inversion_variance(K,f_number+initial_point-1);
        d(f_number)=seismic_data(K,f_number+initial_point-1);
    end
    [inim0,ERRORS(K,:)]=MT2(inim0,inivar,wavelet,d,T0,markovlen,endline);
    result_tmp=zeros(length(inim0_tmp),1);
    for MM=1:uesful_len
        result_tmp(MM+initial_point-1)=inim0(MM);
    end
    saveBestX(K,:)=result_tmp;
 end
 %% ������
saveBestX=saveBestX';
dicSavePath = sprintf('./ResultsData_simple_richer/add_syn/MCMCwell_39_trace_68/markovlen%d_T0%fsaveBestX1.mat', markovlen, T0);
save(dicSavePath, 'saveBestX');