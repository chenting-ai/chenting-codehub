%% 利用marmousi-SM模型合成地震记录，生成初始模型
clc;
clear all;


public_path='E:\matlab_project\The_stochastic_inversion\NewDictionary_MCMC_inversion_move4\marmousi_SM\';
model_path=[public_path,'usefuldata\marmousi2_SM.mat'];%marmousi2_SM
load(model_path);
%子波
waveletFreq = 30;
wave = s_create_wavelet({'type','ricker'}, {'frequencies',waveletFreq}, {'step', 4}, {'wlength',200});
wavelet = wave.traces*100000;

%% 合成地震记录

trueModel = marmousi2_SM;
sampNum = size(trueModel, 1);
[marmousi_SM_syn, G] = csGenPost1DForwardModel(wavelet, trueModel, sampNum, sampNum );


noiseType = 1;
noiseDB = 4;
noiseOption = {'gs','lp'};
marmousi_SM_syn_4dB = wyjGenNormNoise2D(marmousi_SM_syn, noiseDB, noiseType);

%% 生成初始模型


Wn = 0.1;
marmousi_SM_init = csFilterProfile(trueModel, Wn, 'v'); % 纵向滤波
marmousi_SM_init = csFilterProfile(marmousi_SM_init, Wn, 'h'); % 横向滤波

[marmousi_SM_syn_init, G] = csGenPost1DForwardModel(wavelet, marmousi_SM_init, sampNum, sampNum );








