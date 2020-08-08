[seismic,bic_header,binary_header]=read_segy_file('./20200310/vp_vs-HLF-direct-20200402-HRS-model-CSR-90Hz-K-1-DIC-2T1_T3resample.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
Impedence_data=seismic.traces;
wellpoint2=200;
temp_Impedence_data=zeros(wellpoint2,size(Impedence_data,2));
point_number=size(Impedence_data,1);
for i=1:size(Impedence_data,2)
    up=1;
    down=1;
    for j=1:point_number
        if Impedence_data(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up+10:point_number
        if Impedence_data(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len=down-up+1;
        % 插值，将地震数据插值成井数据一样的点数 
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint2);
    seismic_val=interp1(useful_len_seismic,Impedence_data(up:down,i),inter_useful_len_seismic,'spline');
    temp_Impedence_data(:,i)=seismic_val;
end
temp_Impedence_data3=reshape(temp_Impedence_data,[wellpoint2,631,501]);
if 1 
    Wn=0.1;
    for i=1:501
        temp_Impedence_data22(:,:,i) = csFilterProfile(temp_Impedence_data3(:,:,i), Wn, 'h');   % 纵向滤波  
    end
    for i=1:631
        temp_Impedence_data22(:,i,:) = csFilterProfile(temp_Impedence_data22(:,i,:), Wn, 'h');   % 纵向滤波  
    end
end
temp_Impedence_data222=reshape(temp_Impedence_data22,[wellpoint2,631*501]);
for i=1:size(Impedence_data,2)
    up=1;
    down=1;
    for j=1:point_number
        if Impedence_data(j,i)~=0
            up=j;
            break;
        end 
    end
    for k=up+10:point_number
        if Impedence_data(k,i)==0
            down=k-1;
            break;
        end 
    end
    useful_len=down-up+1;
        % 插值，将地震数据插值成井数据一样的点数 
    useful_len_seismic = linspace(1,useful_len,useful_len);
    inter_useful_len_seismic = linspace(1,useful_len,wellpoint2);
    seismic_val=interp1(inter_useful_len_seismic,temp_Impedence_data222(:,i),useful_len_seismic,'spline');
    Impedence_data(up:down,i)=seismic_val;
end
seismic.traces=Impedence_data;
write_segy_file(seismic,'./20200310/vp_vs-HLF-direct-20200402-HRS-model-CSR-90Hz-K-1-DIC-2T1_T3resamplefilter0.1.sgy',{'headers',{'iline_no',189,4},{'xline_no',193,4}});   %写地震数据

