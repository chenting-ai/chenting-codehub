clc;
clear all;

%% ��ȡ�ļ����������
%%%�ļ��У���ʼ���迹ģ�ͣ��������ݣ��������ݣ��Ӳ�
%%%�����У�
T0=20;    %%%��ʼ�¶�
markovlen=5;
endline=0.1;
Number_well=52;
%��ʱд��  ��ȡ�ļ�д�ñȽϼ�
wavelet=load('.\SeismicData\wavelet1225.txt');                      %%load wavelet
if Number_well==26
    impedence=load('.\SeismicData\inline99_initalmodel_26.mat');    %%load varance
    seismic_data=load('.\SeismicData\inline99_seismic.txt');        %%load seismic
    variance=load('.\SeismicData\inline99_varance_26.mat');         %%load inital impendence
    seismic_data=seismic_data(1:2001,1:100);
    initial_impedence=impedence.initial_impedence;  
    inversion_variance=variance.inversion_variance;  
end

if Number_well==52
    impedence=load('.\SeismicData\Xline76_well52initial.mat');    %%load varance
    seismic_data=load('.\SeismicData\Xline76_seismic.mat');        %%load seismic
    variance=load('.\SeismicData\Xline76_well52var.mat');         %%load inital impendence
    seismic_data=seismic_data.Xline76_seismic;
    initial_impedence=impedence.Xline76_well52initial;  
    inversion_variance=variance.Xline76_well52var;  
end

if Number_well==104
    inversion_variance=load('.\SeismicData\inline99_varance_104.txt');   %%load varance
    seismic_data=load('.\SeismicData\inline99_seismic.txt');             %%load seismic
    initial_impedence=load('.\SeismicData\inline99_initalmodel_104.txt');%%load inital impendence
    seismic_data=seismic_data(:,1:100);
    initial_impedence=initial_impedence(:,1:100);
    inversion_variance=inversion_variance(:,1:100);
end


seismic_data=seismic_data';
initial_impedence=initial_impedence';
inversion_variance=inversion_variance';

%�����Ž���ľ����С
[row, col]=size(initial_impedence);
saveBestX=zeros(row,col);
% ERRORS=zeros(row,:);
%% ��������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���м���
% CoreNum=5; %���õĴ���������5
% if matlabpool('size')<=0  %֮ǰû�д�
%     matlabpool('open','local',CoreNum);
% else  %֮ǰ�Ѿ���
%     disp('matlab pool already started');
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for K=1:row                          %%%�������
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
MCMC_800=saveBestX;
imagesc(saveBestX);
dicSavePath = sprintf('./20200320result/MCMC/markovlen%d_T0%fsaveBestX1.mat', markovlen, T0);
save(dicSavePath, 'saveBestX');