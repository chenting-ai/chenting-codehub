%% 生成初始剖面的地震记录及偏移情况
% markovlen=5;
% T0=20;
well_number=52;
dicSavePath1 = sprintf('./ResultsData52_nohorizon_move/MTDL2_1_25_2340/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52_25_2340.mat',markovlen,T0);
impedence = load(dicSavePath1);      %%加载初始波阻抗
impedence=impedence.saveBestX;  
dicSavePath2 = sprintf('./well_plot/sparsitythres20_Dictionary/dictionarydataWell_Num52_2M_25L_2340.mat');
temple_DIC = load(dicSavePath2);                %%load dictionaary
DIC = temple_DIC.D2;
DIW = temple_DIC.W2;
wavelet=load('.\SeismicData\wavelet1225.txt'); 
name1 = sprintf('52井合成地震记录');  
name2 = sprintf('52井合最终偏移量'); 
path = sprintf('F:/matlab临时图片存储/第三轮/52口井初始剖面');

move=zeros(size(impedence));
synseismic=zeros(size(impedence));
[Atomic_length,~]=size(DIC);
for i=1:100
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
    for j=1:uesful_len
        if j>uesful_len-Atomic_length/2
           jj=uesful_len-Atomic_length/2;
        elseif j<Atomic_length/2
           jj=Atomic_length/2;
        else
           jj=j;
        end
        temp_data2=[inim1(jj-Atomic_length/2+1:jj+Atomic_length/2)];
        Spare2=omp(DIC,temp_data2,DIC'*DIC,20);  
        temp_Hor=[];
        temp_Hor=[temp_Hor;abs(j-15);abs(j-20);abs(j-25);abs(j-30);abs(j-35);abs(j-40);abs(j-45);abs(j-50);abs(j-55)];
        [~,Hor] = min(temp_Hor); 
        score_est =  DIW * Spare2;
        [~, maxind_est] = max(score_est);  % classifying
        move(initial_point+j-1,i)=abs(maxind_est-Hor);
        
        
        reflect=zeros(uesful_len-1,1);           %%%
        for l=1:uesful_len-1                     %%%reflect
            reflect(l)=(impedence(initial_point+l,i)-impedence(initial_point+l-1,i))/(impedence(initial_point+l,i)+impedence(initial_point+l-1,i));
        end
        syn_seis=conv(wavelet,reflect);         %%%
        lensyn_seis=length(syn_seis);           %%%length of syn_seis
        syn_seis=syn_seis(round((lensyn_seis-uesful_len)/2+2):round((lensyn_seis-uesful_len)/2+uesful_len+1));
        synseismic(initial_point:termination_point,i) = syn_seis;  
        
        
    end
end
dicSavePath3 = sprintf('./ResultsData52_nohorizon_move/MTDL2_1_25_2340/markovlen%d_T0%f_move.mat',markovlen,T0);
save(dicSavePath3, 'move');
dicSavePath4 = sprintf('./ResultsData52_nohorizon_move/MTDL2_1_25_2340/markovlen%d_T0%f_synseismic.mat',markovlen,T0);
save(dicSavePath4, 'synseismic');








