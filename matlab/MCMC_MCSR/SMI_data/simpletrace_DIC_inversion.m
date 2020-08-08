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
wavelet=load('.\SeismicData\wavelet1225.txt');      

%% 检测有效数据长度
for sampling_number=1:2001      
    if( well(sampling_number,1)~=0)
        initial_point=sampling_number;
        break;
    end
end
for sampling_number=sampling_number:2001   
    if( well(sampling_number,1 )~=0)
       termination_point = sampling_number;
    end
end
useful_len = termination_point-initial_point+1;
trueModel = well(initial_point:termination_point,1);


%% 设置参数
% 输出符号信息
typeMethods = {'GD', 'BFGS', 'CG', 'NCG', 'VSSNSGA', 'csNCG'};  % 迭代方法(1-梯度下降 2-拟牛顿法 3-共轭梯度法 4-厍斌拟共轭梯度 5-符号梯度法 6-自适应拟共轭)
typeObj = {'L1', 'L2', 'huber', 'MCC','MCCi', 'SAM', 'GEV'};  % 目标函数(1-L1,2-L2,3-huber,4-我写的Mcc,5-厍斌写的Mcci,6-混合范数,7-广义极值范数)    
typeReg = {'none', 'Tic1', 'TV', 'MGS'};
testItems = {
 % 迭代算法，目标函数，正则项，正则化参数lamda,迭代次数，MGS的sigma值，MCC的theta值
     [4 2 2 5  30     0.0001 0.001];  %Tikhonov
%      [4 2 3 0.05  150     0.0001 0.001]; %TV
     
     
    };

color = {'r', 'k', 'b--', 'g', 'm--', 'y', 'b', 'r--', 'g--',  'k--'};
tstr = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)'};


%% 正演
% [poststack, G] = csGenPost1DForwardModel(wavelet, trueModel, useful_len, useful_len-1 );
% 添加噪声
% noiseType = 1;
% noiseDB = 4;
% noiseOption = {'gs','lp'};
% postNoise = wyjGenNormNoise2D(poststack, noiseDB, noiseType);
%纵向滤波作为初始模型
Wn = 0.03;
initModel = csFilterProfile(trueModel, Wn, 'v'); % 纵向滤波
% u0 = log(initModel);


%% 开始反演
result = zeros(useful_len, 1);
% res = zeros(testItems{1}(5),1);

 
    fprintf('trace：%d\n',35);
    inter=1;
    Atomic_length=30;
G=generalGD(wavelet,[],Atomic_length,Atomic_length-1);
    
    
    
for j=1:inter:useful_len-Atomic_length+1 
    poststack=G*trueModel(j:j+Atomic_length-1,1);
    u0 = log(initModel(j:j+Atomic_length-1,1));
    [temp_spareu1, res] = csLinearSolver(poststack,      [],         G,     u0,     [],    testItems{1});       
                                               %    噪声记录    真实记录   系统矩阵  初始模型         真实模型           
    result(j:j+Atomic_length-1,1) = exp(temp_spareu1);                                         
end                                                                                                                  
dicSavePath = sprintf('./ResultsData_simple/general_inversion/generalinversion2.mat');
save(dicSavePath, 'result');






