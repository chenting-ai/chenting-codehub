clc;
clear all;

%% ��ȡ�ļ����������
%%%�ļ��У���ʼ���迹ģ�ͣ��������ݣ��������ݣ��Ӳ�
%%%�����У�
T0=100;    %%%��ʼ�¶�
markovlen=20;
endline=0.001;

%��ʱд��  ��ȡ�ļ�д�ñȽϼ�
public_path='E:\matlab_project\The_stochastic_inversion\NewDictionary_MCMC_inversion_move4\marmousi_SM\';
wavelet_path=[public_path,'usefuldata\wavelet_10000.mat'];%marmousi2_SM
impedence_path=[public_path,'usefuldata\marmousi_SM_init_0.1.mat'];%marmousi2_SM
seismic_path=[public_path,'usefuldata\marmousi_SM_syn_10000.mat'];%marmousi2_SM

wavelet=load(wavelet_path);             %%load wavelet
impedence=load(impedence_path);         %%load varance
seismic_data=load(seismic_path);        %%load seismic

wavelet=wavelet.wavelet;
seismic_data=seismic_data.marmousi_SM_syn;
initial_impedence=impedence.marmousi_SM_init;  

seismic_data=seismic_data';
initial_impedence=initial_impedence';

%�����Ž���ľ����С
[row, col]=size(initial_impedence);
saveBestX=zeros(row,col);
% ERRORS=zeros(row,:);
%% ��������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���м���
CoreNum=5; %���õĴ���������5
if matlabpool('size')<=0  %֮ǰû�д�
    matlabpool('open','local',CoreNum);
else  %֮ǰ�Ѿ���
    disp('matlab pool already started');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% h=waitbar(0,'please wait');
parfor K=1:row           %%%�������
%     str=['1������...',num2str(K/row*100),'%'];
%     waitbar(K/row,h,str)
    fprintf('������%d��\n',K);
    inim0_tmp=initial_impedence(K,:);  %%%��ʼ���迹
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
        inivar(f_number)=1000;
        d(f_number)=seismic_data(K,f_number+initial_point-1);
    end
    [inim0,~]=MT2_marmousi_SM(inim0,inivar,wavelet,d,T0,markovlen,endline);
    result_tmp=zeros(length(inim0_tmp),1);
    for MM=1:uesful_len
        result_tmp(MM+initial_point-1)=inim0(MM);
    end
    saveBestX(K,:)=result_tmp;
end
%% ������
saveBestX=saveBestX';
% imagesc(saveBestX);
dicSavePath = sprintf('%sresult/MCMC/markovlen%d_T0%fsaveBestX1_inlimit_2.mat',public_path, markovlen, T0);
save(dicSavePath, 'saveBestX');