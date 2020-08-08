clc;
clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%
%% load data
wavelet=load('.\SeismicData\wavelet1225.txt');                                             %%load wavelet
initial_impedence=load('./ResultsData104/MCMC/markovlen10_T020.000000saveBestX1.mat' );    %%load inital model
% initial_impedence=load('.\SeismicData\polldata.txt');                                      %%load inital impendence
seismic_data=load('.\SeismicData\pollseis.txt');                                           %%load seismic
temple_DIC=load('.\Dictionaries\logMCMCDIC_30_1500_NWell_100_filt_1.00.mat');              %%load dictionaary

seismic_data=seismic_data(:,1:100);
initial_impedence=initial_impedence.saveBestX(:,1:100);
maxvalwave=max(wavelet);
wavelet=wavelet/maxvalwave;
DIC=temple_DIC.DIC;

[sampNum,traceNum] = size(initial_impedence);
[Atomic_length,AtomicNum]=size(DIC);
%% ���ؾ�����
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%���ز⾮ֵ
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%���ز⾮ʱ�����
well=zeros(2001,1);
K1=39;
for i=1:503
    if Well_timeData(i,K1)==fix(Well_timeData(i,K1))&&Well_timeData(i,K1)~=0
        well(fix(Well_timeData(i,K1)),1)=Well_valueData(i,K1);
    end
end
well(805,1)=0;
well(1035,1)=0;
trueModel=well;
%% ���ò���
% ���������Ϣ
typeMethods = {'GD', 'BFGS', 'CG', 'NCG', 'VSSNSGA', 'csNCG'};  % ��������(1-�ݶ��½� 2-��ţ�ٷ� 3-�����ݶȷ� 4-�Ǳ��⹲���ݶ� 5-�����ݶȷ� 6-����Ӧ�⹲��)
typeObj = {'L1', 'L2', 'huber', 'MCC','MCCi', 'SAM', 'GEV'};  % Ŀ�꺯��(1-L1,2-L2,3-huber,4-��д��Mcc,5-�Ǳ�д��Mcci,6-��Ϸ���,7-���弫ֵ����)    
typeReg = {'none', 'Tic1', 'TV', 'MGS'};
testItems = {
 % �����㷨��Ŀ�꺯������������򻯲���lamda,����������MGS��sigmaֵ��MCC��thetaֵ
     [4 1 3 5  30     0.0001 0.001];  %Tikhonov
%      [4 2 3 0.05  150     0.0001 0.001]; %TV
     
     
    };

color = {'r', 'k', 'b--', 'g', 'm--', 'y', 'b', 'r--', 'g--',  'k--'};
tstr = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)'};
%% ����
% [poststack, G] = csGenPost1DForwardModel(wavelet, trueModel, useful_len, useful_len-1 );
% �������
noiseType = 1;
noiseDB = 40;
noiseOption = {'gs','lp'};
% postNoise = wyjGenNormNoise2D(poststack, noiseDB, noiseType);
%�����˲���Ϊ��ʼģ��
Wn = 0.03;
% initModel = csFilterProfile(trueModel, Wn, 'v'); % �����˲�
% u0 = log(initModel);
%%������������GD

% GD=generalGD(wavelet,DIC,Atomic_length,Atomic_length);

%% ��ʼ����
result = zeros(sampNum, traceNum);
res = zeros(testItems{1}(5),traceNum);

for i = 68 
    fprintf('trace��%d\n',i);
    inter=1;
    
    for sampling_number=1:sampNum      
        if( initial_impedence(sampling_number,i)~=0)
            initial_point=sampling_number;
            break;
        end
    end
    for sampling_number=sampling_number:sampNum   
        if( initial_impedence(sampling_number,i )~=0)
           termination_point=sampling_number;
        end
    end
    initModel0 = csFilterProfile(trueModel, Wn, 'v'); % �����˲�
    useful_len=termination_point-initial_point+1;
    
    if 1 %%���淴��
        for j=initial_point:inter:termination_point-Atomic_length+1 
            [poststack, G] = csGenPost1DForwardModel(wavelet, trueModel(j:j+Atomic_length-1,1), Atomic_length,Atomic_length-1 );
            GD = G*DIC;

            tempu0 = log(initModel0(j:j+Atomic_length-1,1)); 
    %         tempu0 = log(initial_impedence(j:j+Atomic_length-1,i));
            temp_spareu0=omp(DIC,tempu0,DIC'*DIC,30);
            tempd  = poststack;%(j:j+Atomic_length-1,i);

        [temp_spareu1, res(:,i)] = csLinearSolver(tempd,      [],         GD,     temp_spareu0,     [],    testItems{1});       
                                                   %    ������¼    ��ʵ��¼   ϵͳ����  ��ʼģ��         ��ʵģ��           
        result(j:j+Atomic_length-1,i) = exp(DIC*temp_spareu1);                                         
        end
    end
    
    if 0 %%ϡ�跴��
        % ����ѡ��
        sparseMethod = {
            % �ֵ�ǿ�� Ϊ0��ʾ��ȫ���ֵ��ؽ� 

             % 1���� ����  0���������
             %   ϡ���   ��������    �������    �ؽ���ֵĴ�С    ��������    �Ƿ��ؽ�   �Ƿ�Ϊ�õ����ķ������  
            [ 0     1       1          0.1          2              100         1              1];
             %   ϡ���   ��������    �ֵ�����    ��ʼģ������      ��������    �Ƿ��ؽ�   �Ƿ�Ϊ�õ����ķ������  
        %     [0      1         1          0.1         0.005            8         1               0];
        };
        global globalParam;
        globalParam.convergenceModel = 1;
        % sparseItem = size(sparseMethod, 1);
        % sparseResult = size(sampNum, sparseItem);
        global GSparseInvParam;
        GSparseInvParam.sizeAtom = 30;              % ԭ�Ӵ�С
        GSparseInvParam.nAtom = 1500;                % ԭ�Ӹ���
        GSparseInvParam.DIC = DIC;
        GSparseInvParam.xIterNum = 20;

%         stpShowDictionary(DIC);

        for j=initial_point:inter:termination_point-Atomic_length+1 
            [poststack, G] = csGenPost1DForwardModel(wavelet, trueModel(j:j+Atomic_length-1,1), Atomic_length,Atomic_length-1 );
            maxVal = max(max(abs(poststack)));
            
           
            param = sparseMethod{1};
            stpInitSparseInvParam_post(param, DIC, Atomic_length, G/maxVal);
            % RMSES = zeros(sparseItem,GSparseInvParam.iterNum );
%             u0 = log(initial_impedence(j:j+Atomic_length-1,i ));
            u0 = log(initModel0(j:j+Atomic_length-1,1 ));
            if GSparseInvParam.isIterative
                [sparseResult, RMSES] = stpSparseInversion_post(poststack/maxVal, G/maxVal, u0, log(trueModel(j:j+Atomic_length-1,1)));
            else
                [sparseResult, RMSES] = stpSparseInversion_post_fast(poststack/maxVal, G/maxVal, u0, log(trueModel(j:j+Atomic_length-1,1)));
            end
            result(j:j+Atomic_length-1,i) = exp(sparseResult);         
        end
        figure;
        t=1:useful_len;

        plot(trueModel(initial_point:termination_point,1)/1000, t, 'b--', 'linewidth', 1.5); hold on;
        plot(initModel0(initial_point:termination_point,1)/1000, t, 'g', 'linewidth', 1.5);hold on;  
        plot(result(initial_point:termination_point,i)/1000, t, 'r', 'linewidth', 1.5); 
        set(gcf, 'position', [100 100 250 800]);

        legend('���ݽ��', '��ʵģ��', '��ʼģ��');
        set(gca, 'xlim', [min(trueModel(initial_point:termination_point,1)/1000)-1.5 max(trueModel(initial_point:termination_point,1)/1000)+1.5]);
        set(gca, 'ylim', [0 t(end)]);
        set(gca, 'ydir','reverse');
    %         set(gca,'YTick', 20:40:useful_len);
        set(gca,'YTickLabel',initial_point:10:termination_point);
  
    end
    T=8;
    if 0  %%omp�㷨
        for j=initial_point:inter:termination_point-Atomic_length+1 
            [poststack, G] = csGenPost1DForwardModel(wavelet, trueModel(j:j+Atomic_length-1,1), Atomic_length,Atomic_length-1 );
%             maxVal = max(max(abs(poststack)));
            tempGD=G*DIC;
            GD=tempGD*spdiag(1./sqrt(sum(tempGD.*tempGD)));
            
            [sparseResult] = omp(GD, poststack, GD'*GD, T);

            result(j:j+Atomic_length-1,i) = exp(DIC*spdiag(1./sqrt(sum(tempGD.*tempGD)))*sparseResult);         
        end
%         invG=pinv(G);
%         test1=log(trueModel(j:j+Atomic_length-1,1));
%         test2=invG*poststack;
%         a=G*DIC*spdiag(1./sqrt(sum(tempGD.*tempGD)))*sparseResult;
%         b=G*log(trueModel(j:j+Atomic_length-1,1));
        figure;
        t=1:useful_len;

        plot(trueModel(initial_point:termination_point,1)/1000, t, 'b--', 'linewidth', 1.5); hold on;
        plot(result(initial_point:termination_point,i)/1000, t, 'r', 'linewidth', 1.5); 
        set(gcf, 'position', [100 100 250 800]);

        legend('��ʵģ��', '���ݽ��');
        set(gca, 'xlim', [min(trueModel(initial_point:termination_point,1)/1000)-1.5 max(trueModel(initial_point:termination_point,1)/1000)+1.5]);
        set(gca, 'ylim', [0 t(end)]);
        set(gca, 'ydir','reverse');
        set(gca,'YTickLabel',initial_point:10:termination_point);
    end
    
    
    
end

dicSavePath = sprintf('./ResultsData_simple/general_inversion/generalinversion3.mat');
save(dicSavePath, 'result');













