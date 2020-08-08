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
%% 加载井数据
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_valueData.mat');%加载测井值
load('D:\MATLAB\bin\The_stochastic_inversion\Dictionary_inversion0-1\Load_WellData\Well_timeData.mat');%加载测井时间深度
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
%% 设置参数
% 输出符号信息
typeMethods = {'GD', 'BFGS', 'CG', 'NCG', 'VSSNSGA', 'csNCG'};  % 迭代方法(1-梯度下降 2-拟牛顿法 3-共轭梯度法 4-厍斌拟共轭梯度 5-符号梯度法 6-自适应拟共轭)
typeObj = {'L1', 'L2', 'huber', 'MCC','MCCi', 'SAM', 'GEV'};  % 目标函数(1-L1,2-L2,3-huber,4-我写的Mcc,5-厍斌写的Mcci,6-混合范数,7-广义极值范数)    
typeReg = {'none', 'Tic1', 'TV', 'MGS'};
testItems = {
 % 迭代算法，目标函数，正则项，正则化参数lamda,迭代次数，MGS的sigma值，MCC的theta值
     [4 1 3 5  30     0.0001 0.001];  %Tikhonov
%      [4 2 3 0.05  150     0.0001 0.001]; %TV
     
     
    };

color = {'r', 'k', 'b--', 'g', 'm--', 'y', 'b', 'r--', 'g--',  'k--'};
tstr = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)'};
%% 正演
% [poststack, G] = csGenPost1DForwardModel(wavelet, trueModel, useful_len, useful_len-1 );
% 添加噪声
noiseType = 1;
noiseDB = 40;
noiseOption = {'gs','lp'};
% postNoise = wyjGenNormNoise2D(poststack, noiseDB, noiseType);
%纵向滤波作为初始模型
Wn = 0.03;
% initModel = csFilterProfile(trueModel, Wn, 'v'); % 纵向滤波
% u0 = log(initModel);
%%生成正演算子GD

% GD=generalGD(wavelet,DIC,Atomic_length,Atomic_length);

%% 开始反演
result = zeros(sampNum, traceNum);
res = zeros(testItems{1}(5),traceNum);

for i = 68 
    fprintf('trace：%d\n',i);
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
    initModel0 = csFilterProfile(trueModel, Wn, 'v'); % 纵向滤波
    useful_len=termination_point-initial_point+1;
    
    if 1 %%常规反演
        for j=initial_point:inter:termination_point-Atomic_length+1 
            [poststack, G] = csGenPost1DForwardModel(wavelet, trueModel(j:j+Atomic_length-1,1), Atomic_length,Atomic_length-1 );
            GD = G*DIC;

            tempu0 = log(initModel0(j:j+Atomic_length-1,1)); 
    %         tempu0 = log(initial_impedence(j:j+Atomic_length-1,i));
            temp_spareu0=omp(DIC,tempu0,DIC'*DIC,30);
            tempd  = poststack;%(j:j+Atomic_length-1,i);

        [temp_spareu1, res(:,i)] = csLinearSolver(tempd,      [],         GD,     temp_spareu0,     [],    testItems{1});       
                                                   %    噪声记录    真实记录   系统矩阵  初始模型         真实模型           
        result(j:j+Atomic_length-1,i) = exp(DIC*temp_spareu1);                                         
        end
    end
    
    if 0 %%稀疏反演
        % 方法选项
        sparseMethod = {
            % 字典强度 为0表示完全用字典重建 

             % 1代表 迭代  0代表解析解
             %   稀疏度   滑动步长    反演组分    重建组分的大小    迭代次数    是否重建   是否为用迭代的方法求解  
            [ 0     1       1          0.1          2              100         1              1];
             %   稀疏度   滑动步长    字典力度    初始模型力度      迭代次数    是否重建   是否为用迭代的方法求解  
        %     [0      1         1          0.1         0.005            8         1               0];
        };
        global globalParam;
        globalParam.convergenceModel = 1;
        % sparseItem = size(sparseMethod, 1);
        % sparseResult = size(sampNum, sparseItem);
        global GSparseInvParam;
        GSparseInvParam.sizeAtom = 30;              % 原子大小
        GSparseInvParam.nAtom = 1500;                % 原子个数
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

        legend('反演结果', '真实模型', '初始模型');
        set(gca, 'xlim', [min(trueModel(initial_point:termination_point,1)/1000)-1.5 max(trueModel(initial_point:termination_point,1)/1000)+1.5]);
        set(gca, 'ylim', [0 t(end)]);
        set(gca, 'ydir','reverse');
    %         set(gca,'YTick', 20:40:useful_len);
        set(gca,'YTickLabel',initial_point:10:termination_point);
  
    end
    T=8;
    if 0  %%omp算法
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

        legend('真实模型', '反演结果');
        set(gca, 'xlim', [min(trueModel(initial_point:termination_point,1)/1000)-1.5 max(trueModel(initial_point:termination_point,1)/1000)+1.5]);
        set(gca, 'ylim', [0 t(end)]);
        set(gca, 'ydir','reverse');
        set(gca,'YTickLabel',initial_point:10:termination_point);
    end
    
    
    
end

dicSavePath = sprintf('./ResultsData_simple/general_inversion/generalinversion3.mat');
save(dicSavePath, 'result');













