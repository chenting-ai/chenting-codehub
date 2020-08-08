[seismic1,bic_header,binary_header]=read_segy_file('./LNdata/vp-HLF-computed-20200402-HRS-model-CSR-90Hz-K-1-DIC-2.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %����������
Impedence_data1=seismic1.traces;
[seismic2,bic_header,binary_header]=read_segy_file('./LNdata/vp-raw-20200402-HRS-model-CSR.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %����������
Impedence_data2=seismic2.traces;
[seismic3,bic_header,binary_header]=read_segy_file('./LNdata/vs-HLF-computed-20200402-HRS-model-CSR-90Hz-K-1-DIC-2.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %����������
Impedence_data3=seismic3.traces;
[seismic4,bic_header,binary_header]=read_segy_file('./LNdata/vs-raw-20200402-HRS-model-CSR.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %����������
Impedence_data4=seismic4.traces;

hor=load('./LNdata/smoothTII0chp - ����.txt');

for i=1:size(Impedence_data1,2)
    
    up=round(hor(i,3))-3400+1-2;
    down=round(hor(i,3))-3400+1+2;
    Impedence_data2(up:down,i)=Impedence_data1(up:down,i);
    Impedence_data4(up:down,i)=Impedence_data3(up:down,i);
end
seismic.traces=Impedence_data2;
write_segy_file(seismic,'./LNdata/vp-raw-20200402-HRS-model-CSR_insert90HzT20.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������
seismic.traces=Impedence_data4;
write_segy_file(seismic,'./LNdata/vs-raw-20200402-HRS-model-CSR_insert90HzT20.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %д��������












