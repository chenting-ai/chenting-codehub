function [ seismic,well ] = loaddata(  )
%LOADDATA 此处显示有关此函数的摘要
%   此处显示详细说明



    % 地震
    seismic_data1=load('.\LNusefuldata\cdp_stack1500offset1msT1_50_T2smooth.mat');    
    seismic=seismic_data1.cdp_stack1500offset1msT1_50_T2smooth;
    % 井数据
    load('.\LNusefuldata\well_data_new0323.mat');%加载测井值
    well=well_data_new0323;
end

