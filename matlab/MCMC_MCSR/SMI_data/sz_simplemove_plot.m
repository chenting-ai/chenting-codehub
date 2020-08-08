clear
clc


T0=20;
markovlen=20;
well_number=52;
load('.\SeismicData\inline99_initalmodel_52.mat');%initial_impedence
dicSavePath1 = sprintf('./ResultsData%d_horizon_move/MTDL2_1/markovlen%d_T0%f_move.mat',well_number, markovlen, T0);
saveBestX1 = load(dicSavePath1);      %%加载结果
move=saveBestX1.move;


dicSavePath2 = sprintf('./ResultsData%d_nohorizon_move/MTDL2_1/markovlen%d_T0%f_move.mat',well_number, markovlen, T0);
saveBestX2 = load(dicSavePath2);      %%加载结果
move2=saveBestX2.move;

dicSavePath3 = sprintf('./ResultsData%d_nohorizon_move/MCMC/markovlen%d_T0%f_move.mat',well_number, markovlen, T0);
saveBestX3 = load(dicSavePath3);      %%加载结果
move3=saveBestX3.move;

for i=1:100
    
    for sampling_number=1:2001      
    if( initial_impedence(sampling_number,i)~=0)
        initial_point=sampling_number;
        break;
    end
    end
    for sampling_number=sampling_number:2001   
        if( initial_impedence(sampling_number,i )~=0)
           termination_point = sampling_number;
        end
    end
    uesful_len=termination_point-initial_point+1;    %%%useful length
    inim1 = move(initial_point:termination_point,i);
    inim2 = move2(initial_point:termination_point,i);
    inim3 = move3(initial_point:termination_point,i);
    count1=0;
    count2=0;
    count3=0;
    for j=13:uesful_len-13
        if inim1(j)<3
            count1=count1+1;
        end
        if inim2(j)<3
            count2=count2+1;
        end
        if inim3(j)<3
            count3=count3+1;
        end
        
    end
    move_corect1(i)=count1/uesful_len;
    move_corect2(i)=count2/uesful_len;
    move_corect3(i)=count3/uesful_len;
end


% move_sum3=sum(move3);
% move_sum2=sum(move2);
% move_sum=sum(move);

trace_number=1:100;
linewidth=1.5;

plot(trace_number,move_corect1  , 'r', 'linewidth', linewidth);hold on;
plot(trace_number,move_corect2  , 'k', 'linewidth', linewidth);hold on;
plot(trace_number,move_corect3  , 'g--', 'linewidth', linewidth);hold on;