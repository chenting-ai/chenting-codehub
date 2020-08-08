%% �������� T2��
T2=load('hrs_horizon_TII - ����.txt');
[seismic,bic_header,binary_header]=read_segy_file('./20200310/vp_vs-HLF-direct-20200329-CSR-MATLAB-model-1ms-radon-90Hz-K-1-DIC-1.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %�����������
seismic=seismic.traces;%% ��ʼʱ��3000ms���ü��1ms
trace_number=size(T2,1);
%% Ѱ��T2�����AƮ10ms����Сֵ����¼ʱ�����Ϊ��λ��
T2_f=zeros(size(T2));
T2_f(:,1:2)=T2(:,1:2);
T2_fvalue=zeros(size(T2));
T2_fvalue(:,1:2)=T2(:,1:2);
T2_fvalue_avg=zeros(size(T2));
T2_fvalue_avg(:,1:2)=T2(:,1:2);

for i=1:trace_number
    horizion_up=round(T2(i,3))-10-2999;
    horizion_down=round(T2(i,3))-2999;
    up=1;
    down=1;
    [value,idx]=min(seismic(horizion_up:horizion_down,i));
    T2_f(i,3)=idx+horizion_up+2999-1;
    T2_fvalue(i,3)=value;
    sumvalue=0;
    for k=1:5
        [value,idx]=min(seismic(horizion_up:horizion_down,i));
        seismic(horizion_up+idx-1,i)=5;
        sumvalue=sumvalue+value/5;
    end
    T2_fvalue_avg(i,3)=sumvalue;  
end



%% �������
Wn=0.2;
T2_f(:,3) = reshape(csFilterProfile(reshape(T2_f(:,3),[631,501]), Wn, 'h'),[631*501,1]);   % �����˲�  
T2_f(:,3) = reshape(csFilterProfile(reshape(T2_f(:,3),[631,501]), Wn, 'v'),[631*501,1]);   % �����˲�

save('T2_f2.txt','T2_f','-ascii') ;
save('T2_fvalue2.txt','T2_fvalue','-ascii') ;
save('T2_fvalue_avg2.txt','T2_fvalue_avg','-ascii') ;