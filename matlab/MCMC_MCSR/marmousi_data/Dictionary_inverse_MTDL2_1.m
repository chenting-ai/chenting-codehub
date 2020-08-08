clc;
clear all;
%% load and define
%%file
%%parameter
T0=100;   
markovlen=15;
endline=0.001;


%% load data
public_path='E:\matlab_project\The_stochastic_inversion\NewDictionary_MCMC_inversion_move4\marmousi_SM\';
wavelet_path=[public_path,'usefuldata\wavelet_10000.mat'];%marmousi2_SM
impedence_path=[public_path,'usefuldata\marmousi_SM_init_0.1.mat'];%marmousi2_SM
seismic_path=[public_path,'usefuldata\marmousi_SM_syn_10000.mat'];%marmousi2_SM
well_path=[public_path,'Dictionaries\marmousi14DIC_30_1500_NWell_14_filt_0.70.mat'];%marmousi2_SM
temple_DIC=load(well_path);   %%DIC
wavelet=load(wavelet_path);             %%load wavelet
impedence=load(impedence_path);         %%load varance
seismic_data=load(seismic_path);        %%load seismic

wavelet=wavelet.wavelet;
seismic_data=seismic_data.marmousi_SM_syn;
initial_impedence=impedence.marmousi_SM_init;  


DIC=temple_DIC.DIC;
%% Define the size of the matrix to store the results
[sampNum, traceNum]=size(initial_impedence);
saveBestX=zeros(sampNum, traceNum);
% ERRORS=zeros(1,traceNum);


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
for K=201         
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
        inivar(f_number)=1000;
        d_seismic(f_number)=seismic_data(f_number+initial_point-1,K);
    end
    [inim0,~]=MT_DL2_1modify_marmousi_SM_2(inim0,inivar,wavelet,d_seismic,T0,markovlen,DIC,endline);
    result_tmp=zeros(length(inim0_tmp),1);
    for MM=1:uesful_len
        result_tmp(MM+initial_point-1)=inim0(MM);
    end
    saveBestX(:,K)=result_tmp;
end
%% Êä³ö½á¹û
dicSavePath = sprintf('%sresult/MCDL/markovlen%d_T0%fsaveBestX1trace201_inlimit_2_2.mat',public_path, markovlen, T0);
save(dicSavePath, 'saveBestX');