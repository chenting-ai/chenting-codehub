function [ seismic,well ] = loaddata(  )
%LOADDATA �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��



    % ����
    seismic_data1=load('.\LNusefuldata\cdp_stack1500offset1msT1_50_T2smooth.mat');    
    seismic=seismic_data1.cdp_stack1500offset1msT1_50_T2smooth;
    % ������
    load('.\LNusefuldata\well_data_new0323.mat');%���ز⾮ֵ
    well=well_data_new0323;
end

