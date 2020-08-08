%%%%��ͼ

T0=20;
markovlen=5;
dicSavePath1 = sprintf('./20200320result/MTDL2_1_50_2000/markovlen5_T020.000000_synseismic.mat');
synseismic=load(dicSavePath1);      %%���ؽ��
name= sprintf('T0_%dmarkovlen%d�ϳɵ����¼',T0, markovlen);  
path = sprintf('F:/matlab��ʱͼƬ�洢/����������/MCMC');
synseismic=synseismic.synseismic(800:1050,1:142);
%% ����������
test_synseismic=zeros(100,size(synseismic,2));
for k=1:size(synseismic,2)

        test_synseismic(:,k)=synseismic(95:194,k);


end




figure('color',[1 1 1])
set(gcf, 'position', [100 500 1100 400]);
wigb(test_synseismic);%%���ݲ��迹
set(gca, 'ylim', [0 99]);
set(gca, 'xlim', [0 142]);
set(gca,'yTickLabel', 895:10:994);
set(gca,'xTickLabel', 0:20:142);
box on
axis on
grid off
set(gca,'xaxislocation','top');%��X�����ͼ���ϲ�
ylabel('Time(ms)');
xlabel('trace number');
fileName = sprintf('%s/%s.tif', path, name);
% print('-dtiff', '-r600', fileName);

