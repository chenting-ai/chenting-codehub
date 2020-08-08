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
wavelet=load('.\SeismicData\wavelet1225.txt');      

%% �����Ч���ݳ���
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


%% ���ò���
% ���������Ϣ
typeMethods = {'GD', 'BFGS', 'CG', 'NCG', 'VSSNSGA', 'csNCG'};  % ��������(1-�ݶ��½� 2-��ţ�ٷ� 3-�����ݶȷ� 4-�Ǳ��⹲���ݶ� 5-�����ݶȷ� 6-����Ӧ�⹲��)
typeObj = {'L1', 'L2', 'huber', 'MCC','MCCi', 'SAM', 'GEV'};  % Ŀ�꺯��(1-L1,2-L2,3-huber,4-��д��Mcc,5-�Ǳ�д��Mcci,6-��Ϸ���,7-���弫ֵ����)    
typeReg = {'none', 'Tic1', 'TV', 'MGS'};
testItems = {
 % �����㷨��Ŀ�꺯������������򻯲���lamda,����������MGS��sigmaֵ��MCC��thetaֵ
     [4 2 2 5  30     0.0001 0.001];  %Tikhonov
%      [4 2 3 0.05  150     0.0001 0.001]; %TV
     
     
    };

color = {'r', 'k', 'b--', 'g', 'm--', 'y', 'b', 'r--', 'g--',  'k--'};
tstr = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)'};


%% ����
% [poststack, G] = csGenPost1DForwardModel(wavelet, trueModel, useful_len, useful_len-1 );
% �������
% noiseType = 1;
% noiseDB = 4;
% noiseOption = {'gs','lp'};
% postNoise = wyjGenNormNoise2D(poststack, noiseDB, noiseType);
%�����˲���Ϊ��ʼģ��
Wn = 0.03;
initModel = csFilterProfile(trueModel, Wn, 'v'); % �����˲�
% u0 = log(initModel);


%% ��ʼ����
result = zeros(useful_len, 1);
% res = zeros(testItems{1}(5),1);

 
    fprintf('trace��%d\n',35);
    inter=1;
    Atomic_length=30;
G=generalGD(wavelet,[],Atomic_length,Atomic_length-1);
    
    
    
for j=1:inter:useful_len-Atomic_length+1 
    poststack=G*trueModel(j:j+Atomic_length-1,1);
    u0 = log(initModel(j:j+Atomic_length-1,1));
    [temp_spareu1, res] = csLinearSolver(poststack,      [],         G,     u0,     [],    testItems{1});       
                                               %    ������¼    ��ʵ��¼   ϵͳ����  ��ʼģ��         ��ʵģ��           
    result(j:j+Atomic_length-1,1) = exp(temp_spareu1);                                         
end                                                                                                                  
dicSavePath = sprintf('./ResultsData_simple/general_inversion/generalinversion2.mat');
save(dicSavePath, 'result');






