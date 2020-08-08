% T0=20;
% markovlen=5;
well_number=52;
dicSavePath1 = sprintf('./ResultsData%d_horizon_move/MTDL2_1_25_2340/markovlen%d_T0%fsaveBestX1_MTDL2_1_well52horizon25_2340.mat',well_number, markovlen, T0);
saveBestX1 = load(dicSavePath1);      %%¼ÓÔØ½á¹û
dicSavePath2 = sprintf('./well_plot/sparsitythres20_Dictionary/dictionarydataWell_Num%d_2M_25L_2340.mat',well_number);
temple_DIC = load(dicSavePath2);                %%load dictionaary
DIC = temple_DIC.D2;
DIW = temple_DIC.W2;
name = sprintf('T0_%dmarkovlen%dMCDL%d×îÖÕÆ«ÒÆÁ¿25_2340',T0, markovlen,well_number);  
path = sprintf('F:/matlabÁÙÊ±Í¼Æ¬´æ´¢/µÚÎåÂÖ/MCDL_2horizon-%d',well_number );

tempsaveBestX1=saveBestX1. saveBestX;
move=zeros(size(tempsaveBestX1));
[Atomic_length,~]=size(DIC);
for i=1:100
    i
    for sampling_number=1:size(tempsaveBestX1,1)      
        if( tempsaveBestX1(sampling_number,i)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:size(tempsaveBestX1,1)      
        if( tempsaveBestX1(sampling_number,i)~=0)
           termination_point=sampling_number;
        end
    end
    uesful_len=termination_point-initial_point+1;    %%%useful length
    inim1 = tempsaveBestX1(initial_point:termination_point,i);
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
    end
end

% figure
% imagesc(move(800:1050,1:100));%%·´ÑÝ²¨×è¿¹
% colorbar
% stpSaveFigure(path, name);

dicSavePath3 = sprintf('./ResultsData%d_horizon_move/MTDL2_1_25_2340/markovlen%d_T0%f_move.mat',well_number, markovlen, T0);
save(dicSavePath3, 'move');