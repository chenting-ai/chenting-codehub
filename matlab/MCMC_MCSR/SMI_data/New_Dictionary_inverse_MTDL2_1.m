clc;
clear all;
%% load and define
%%%file
%%%parameter
T0=20;   
markovlen=10;
endline=0.1;
Number_well=52;
%% load data
wavelet=load('.\SeismicData\wavelet1225.txt');                                       %%load wavelet
if Number_well==26
    impedence=load('.\SeismicData\inline99_initalmodel_26.mat');                     %%load varance
    seismic_data=load('.\SeismicData\inline99_seismic.txt');                         %%load seismic
    variance=load('.\SeismicData\inline99_varance_26.mat');                          %%load inital impendence
    temple_DIC=load('.\Dictionaries\dictionarydataWell_Num26_2.mat');                %%load dictionaary
    seismic_data=seismic_data(1:2001,1:100);
    initial_impedence=impedence.initial_impedence;  
    inversion_variance=variance.inversion_variance;  
end

if Number_well==52
    impedence=load('.\SeismicData\inline99_initalmodel_52.mat');                     %%load varance
    seismic_data=load('.\SeismicData\inline99_seismic.txt');                         %%load seismic
    variance=load('.\SeismicData\inline99_varance_52.mat');                          %%load inital impendence
    temple_DIC=load('.\Dictionaries\dictionarydataWell_Num52_22.mat');                %%load dictionaary
    seismic_data=seismic_data(1:2001,1:100);
    initial_impedence=impedence.initial_impedence;  
    inversion_variance=variance.inversion_variance;  
end

if Number_well==104
    inversion_variance=load('.\SeismicData\inline99_varance_104.txt');              %%load varance
    seismic_data=load('.\SeismicData\inline99_seismic.txt');                        %%load seismic
    initial_impedence=load('.\SeismicData\inline99_initalmodel_104.txt');           %%load inital impendence
    temple_DIC=load('.\Dictionaries\dictionarydataWell_Num104_2.mat');              %%load dictionaary
    seismic_data=seismic_data(:,1:100);
    initial_impedence=initial_impedence(:,1:100);
    inversion_variance=inversion_variance(:,1:100);
end

DIC=temple_DIC.D2;
DIW=temple_DIC.W2;
%% Define the size of the matrix to store the results
[sampNum, traceNum]=size(initial_impedence);
saveBestX=zeros(sampNum, traceNum);
atomic_move1=zeros(sampNum, traceNum);
% ERRORS=zeros(1,traceNum);
syn_seis=zeros(sampNum, traceNum);

%% calculate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%par
% CoreNum=5; %the number of cpu 5
% if matlabpool('size')<=0  
%     matlabpool('open','local',CoreNum);
% else  
%     disp('matlab pool already started');
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for K=1:traceNum                       
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
    [inim0,ERRORS(:,K),move1]=MT_DL2_1modify_horizon_move(DIW,inim0,inivar,wavelet,d_seismic,T0,markovlen,DIC,endline);
    result_tmp=zeros(length(inim0_tmp),1);
    atomic_movetemp1=zeros(length(inim0_tmp),1);
    atomic_movetemp2=zeros(length(inim0_tmp),1);
    for MM=1:uesful_len
        result_tmp(MM+initial_point-1)=inim0(MM);
        atomic_movetemp1(MM+initial_point-1)=move1(MM);
    end
    saveBestX(:,K)=result_tmp;
    atomic_move1(:,K)=atomic_movetemp1;
 end
 %% output
dicSavePath = sprintf('./ResultsData%d_horizon_move/MTDL2_1/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52horizon0.9.mat', Number_well,markovlen, T0);
dicSavePath2 = sprintf('./ResultsData%d_horizon_move/MTDL2_1/markovlen%d_T0%fsaveBestX1_ERRORS_MTDL2_1_well52horizon0.9.mat', Number_well,markovlen, T0);
dicSavePath3 = sprintf('./ResultsData%d_horizon_move/MTDL2_1/markovlen%d_T0%fsaveBestX1_move_pro_MTDL2_1_well52horizon0.9.mat',Number_well, markovlen, T0);
save(dicSavePath, 'saveBestX');
save(dicSavePath2, 'ERRORS');
save(dicSavePath3, 'atomic_move1');
imagesc(saveBestX);