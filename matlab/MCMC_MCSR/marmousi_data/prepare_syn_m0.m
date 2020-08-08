%% ����marmousi-SMģ�ͺϳɵ����¼�����ɳ�ʼģ��
clc;
clear all;


public_path='E:\matlab_project\The_stochastic_inversion\NewDictionary_MCMC_inversion_move4\marmousi_SM\';
model_path=[public_path,'usefuldata\marmousi2_SM.mat'];%marmousi2_SM
load(model_path);
%�Ӳ�
waveletFreq = 30;
wave = s_create_wavelet({'type','ricker'}, {'frequencies',waveletFreq}, {'step', 4}, {'wlength',200});
wavelet = wave.traces*100000;

%% �ϳɵ����¼

trueModel = marmousi2_SM;
sampNum = size(trueModel, 1);
[marmousi_SM_syn, G] = csGenPost1DForwardModel(wavelet, trueModel, sampNum, sampNum );


noiseType = 1;
noiseDB = 4;
noiseOption = {'gs','lp'};
marmousi_SM_syn_4dB = wyjGenNormNoise2D(marmousi_SM_syn, noiseDB, noiseType);

%% ���ɳ�ʼģ��


Wn = 0.1;
marmousi_SM_init = csFilterProfile(trueModel, Wn, 'v'); % �����˲�
marmousi_SM_init = csFilterProfile(marmousi_SM_init, Wn, 'h'); % �����˲�

[marmousi_SM_syn_init, G] = csGenPost1DForwardModel(wavelet, marmousi_SM_init, sampNum, sampNum );








