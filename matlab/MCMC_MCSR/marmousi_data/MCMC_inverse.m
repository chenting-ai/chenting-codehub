clc;
clear all;

%% 读取文件及定义参数
%%%文件有：初始波阻抗模型，地震数据，方差数据，子波
%%%参数有：
T0=100;    %%%初始温度
markovlen=20;
endline=0.001;

%临时写的  读取文件写得比较简单
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

%定义存放结果的矩阵大小
[row, col]=size(initial_impedence);
saveBestX=zeros(row,col);
% ERRORS=zeros(row,:);
%% 迭代运算
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%并行计算
CoreNum=5; %调用的处理器个数5
if matlabpool('size')<=0  %之前没有打开
    matlabpool('open','local',CoreNum);
else  %之前已经打开
    disp('matlab pool already started');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% h=waitbar(0,'please wait');
parfor K=1:row           %%%逐道运算
%     str=['1运行中...',num2str(K/row*100),'%'];
%     waitbar(K/row,h,str)
    fprintf('迭代：%d道\n',K);
    inim0_tmp=initial_impedence(K,:);  %%%初始波阻抗
    d_tmp=seismic_data(K,:);           %%%地震数据
    
    for sampling_number=1:length(inim0_tmp)          %%%从非零的点开始
        if(initial_impedence(K,sampling_number)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:length(inim0_tmp)          %%%在最后一个非零结束
        if(initial_impedence(K,sampling_number)~=0)
           termination_point=sampling_number;
        end
    end
    uesful_len=termination_point-initial_point+1;              %%%有效数据长度
    inim0=zeros(uesful_len,1);               %%%波阻抗
    inivar=zeros(uesful_len,1);              %%%方差
    d=zeros(uesful_len,1);
    bestX=zeros(uesful_len,1);               %%%反演结果
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
%% 输出结果
saveBestX=saveBestX';
% imagesc(saveBestX);
dicSavePath = sprintf('%sresult/MCMC/markovlen%d_T0%fsaveBestX1_inlimit_2.mat',public_path, markovlen, T0);
save(dicSavePath, 'saveBestX');