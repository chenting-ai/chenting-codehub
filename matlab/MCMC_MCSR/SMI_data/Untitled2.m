patchs = [trainLogs(1:1+GSparseInvParam.sizeAtom-2,1);700];
patch = [trainLogs(1:1+GSparseInvParam.sizeAtom-2,1);1000];
patchs = [patchs,patch];
t =1:50;
linewidth = 1.5;
figure;set(gcf, 'position', [200 100 220 400]);
plot(patchs(:,1), t, 'k', 'linewidth', linewidth);hold on;
plot(patchs(:,2)+300, t,'r--', 'linewidth', linewidth);hold on;

legend({'700','1000'});
set(gca, 'ydir','reverse');

a=omp(DIC,patchs(:,1),DIC'*DIC,50);
b=omp(DIC,patchs(:,2),DIC'*DIC,50);
aa=DIC*a;
bb=DIC*b;

figure;set(gcf, 'position', [200 100 220 400]);
plot(aa, t, 'k', 'linewidth', linewidth);hold on;
plot(bb+300, t,'r--', 'linewidth', linewidth);hold on;

legend({'700','1000'});
set(gca, 'ydir','reverse');