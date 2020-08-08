T0=20;
markovlen=20;
well_number=52;
dicSavePath1 = sprintf('./ResultsData%d_nohorizon_move/MCMC/markovlen%d_T0%f_move.mat',well_number, markovlen, T0);
saveBestX1 = load(dicSavePath1);      %%加载结果


name= sprintf('T0_%dmarkovlen%d偏移量25_2340',T0,markovlen);
path = sprintf('F:/matlab临时图片存储/第五轮重做/MCMC'); 
move=saveBestX1. move(800:1050,1:100);
test_move=zeros(91,size(move,2));
for k=1:size(move,2)
    if k<40
        test_move(:,k)=move(k+10:k+100,k);   
    elseif k>=40
        test_move(:,k)=move(fix(1.35*k)-4:fix(1.35*k)+86,k);
    end  

end
x=1:size(test_move,1);
y=1:size(test_move,2);
[x,y]=meshgrid(x,y);
x2=1:0.1:size(test_move,1);
y2=1:0.1:size(test_move,2);
[x2,y2]=meshgrid(x2,y2);
z2=interp2(x,y,test_move',x2,y2,'linear');
%%
s_cplot(z2');
set(gcf, 'position', [100 500 500 200]);
% set(gca, 'xlim', [-10 1000]);
set(gca, 'xlim', [0 991]);
set(gca, 'ylim', [0 900]);
set(gca,'yTickLabel', 810:20:900);
set(gca,'xTickLabel', 0:20:100);
caxis([0.99,9]);
% title('反演剖面');
xlabel('trace Number');
ylabel('Time(ms)');
hold on;
fileName = sprintf('%s/%s.tif', path, name);
print('-dtiff', '-r600', fileName);
