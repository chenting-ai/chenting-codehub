clc;
clear all;
%% load and define
%%file
%%parameter
T0=20;   
markovlen=5;
endline=0.1;
Number_well=52;

%% load data
wavelet=load('.\SeismicData\wavelet1225.txt');                                       %%load wavelet
if Number_well==26
    impedence=load('.\SeismicData\inline99_initalmodel_26.mat');                     %%load varance
    seismic_data=load('.\SeismicData\inline99_seismic.txt');                         %%load seismic
    variance=load('.\SeismicData\inline99_varance_26.mat');                          %%load inital impendence
    temple_DIC=load('.\Dictionaries\DIC_50_1500_NWell_52_filt_1.00.mat');     %%load dictionaary
    seismic_data=seismic_data(1:2001,1:100);
    initial_impedence=impedence.initial_impedence;  
    inversion_variance=variance.inversion_variance;  
end

if Number_well==52
    impedence=load('.\SeismicData\Xline76_well52initial.mat');    %%load varance
    seismic_data=load('.\SeismicData\Xline76_seismic.mat');        %%load seismic
    variance=load('.\SeismicData\Xline76_well52var.mat');         %%load inital impendence
    temple_DIC=load('.\Dictionaries\DIC_50_2000_NWell_52_filt_0.70.mat');    %%load dictionaary
    seismic_data=seismic_data.Xline76_seismic;
    initial_impedence=impedence.Xline76_well52initial;  
    inversion_variance=variance.Xline76_well52var;  
end

if Number_well==104
    inversion_variance=load('.\SeismicData\inline99_varance_104.txt');              %%load varance
    seismic_data=load('.\SeismicData\inline99_seismic.txt');                        %%load seismic
    initial_impedence=load('.\SeismicData\inline99_initalmodel_104.txt');           %%load inital impendence
    temple_DIC=load('.\Dictionaries\DIC_50_1500_NWell_208_filt_1.00.mat');   %%load dictionaary
    seismic_data=seismic_data(:,1:100);
    initial_impedence=initial_impedence(:,1:100);
    inversion_variance=inversion_variance(:,1:100);
end

DIC=temple_DIC.DIC;
%% Define the size of the matrix to store the results
[sampNum, traceNum]=size(initial_impedence);
saveBestX=zeros(sampNum, traceNum);
% ERRORS=zeros(1,traceNum);
syn_seis=zeros(sampNum, traceNum);

%% calculate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%par
CoreNum=4; %the number of cpu 5
if matlabpool('size')<=0  
    matlabpool('open','local',CoreNum);
else  
    disp('matlab pool already started');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parfor K=1:traceNum                       
    fprintf('trace£º%d\n',K);
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
    [inim0,ERRORS(:,K)]=MT_DL2_1modify(inim0,inivar,wavelet,d_seismic,T0,markovlen,DIC,endline);
    result_tmp=zeros(length(inim0_tmp),1);
    for MM=1:uesful_len
        result_tmp(MM+initial_point-1)=inim0(MM);
    end
    saveBestX(:,K)=result_tmp;
 end
 %% output
dicSavePath = sprintf('./20200320result/MTDL2_1_50_2000/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52.mat', markovlen, T0);
dicSavePath2 = sprintf('./20200320result/MTDL2_1_50_2000/markovlen%d_T0%fsaveBestX1_ERRORS_MTDL2_1_well52.mat', markovlen, T0);
save(dicSavePath, 'saveBestX');
save(dicSavePath2, 'ERRORS');
imagesc(saveBestX);