%% 生成初始剖面的地震记录及偏移情况
markovlen=5;
T0=20;
well_number=52;
dicSavePath1 = sprintf('./20200320result/MTDL2_1_50_2000/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52.mat',markovlen,T0);
impedence = load(dicSavePath1);      %%加载初始波阻抗
impedence=impedence.saveBestX;  

wavelet=load('.\SeismicData\wavelet1225.txt'); 
name1 = sprintf('52井合成地震记录');  
name2 = sprintf('52井合最终偏移量'); 
path = sprintf('F:/matlab临时图片存储/第三轮/52口井初始剖面');

move=zeros(size(impedence));
synseismic=zeros(size(impedence));
[Atomic_length,~]=size(DIC);
for i=1:142
    i
    for sampling_number=1:size(impedence,1)      
        if( impedence(sampling_number,i)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:size(impedence,1)      
        if( impedence(sampling_number,i)~=0)
           termination_point=sampling_number;
        end
    end
    uesful_len=termination_point-initial_point+1;    %%%useful length
    inim1 = impedence(initial_point:termination_point,i);
        
        
        reflect=zeros(uesful_len-1,1);           %%%
        for l=1:uesful_len-1                     %%%reflect
            reflect(l)=(impedence(initial_point+l,i)-impedence(initial_point+l-1,i))/(impedence(initial_point+l,i)+impedence(initial_point+l-1,i));
        end
        syn_seis=conv(wavelet,reflect);         %%%
        lensyn_seis=length(syn_seis);           %%%length of syn_seis
        syn_seis=syn_seis(round((lensyn_seis-uesful_len)/2+2):round((lensyn_seis-uesful_len)/2+uesful_len+1));
        synseismic(initial_point:termination_point,i) = syn_seis;  
        
end
dicSavePath4 = sprintf('./20200320result/MTDL2_1_50_2000/markovlen%d_T0%f_synseismic.mat',markovlen,T0);
save(dicSavePath4, 'synseismic');








