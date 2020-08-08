

[prediction2,accuracy2,DIC_classfy,W_classfy] = classification2(D2, W2, training_feats, H_train, sparsitythres);
fprintf('\nFinal recognition rate for LC-KSVD2 is : %.03f ', accuracy2);

D22=[DIC_classfy.DIC1,DIC_classfy.DIC2,DIC_classfy.DIC3,DIC_classfy.DIC4,DIC_classfy.DIC5,DIC_classfy.DIC6,DIC_classfy.DIC7,DIC_classfy.DIC8,DIC_classfy.DIC9];
W22=[W_classfy.W1,W_classfy.W2,W_classfy.W3,W_classfy.W4,W_classfy.W5,W_classfy.W6,W_classfy.W7,W_classfy.W8,W_classfy.W9];

[prediction2,accuracy2,err] = classification(D22, W22, training_feats, H_train, sparsitythres);
fprintf('\nFinal recognition rate for LC-KSVD2 is : %.03f ', accuracy2);

stpShowDictionary2(D22);
t=1:70;
linewidth=2;
figure('color',[1 1 1])
plot(well(:,16)/1000-3,t,'k','linewidth',linewidth);hold on;
plot(well(:,31)/1000+2,t,'k','linewidth',linewidth);hold on;
plot(well(:,44)/1000+7,t,'k','linewidth',linewidth);hold on;
plot(well(:,15)/1000+12,t,'k','linewidth',linewidth);hold on;
plot(well(:,37)/1000+17,t,'k','linewidth',linewidth);hold on;
set(gca, 'ydir','reverse');
ylabel('Time(ms)');
set(gca,'YTick', 0:10:70);
set(gca, 'xlim', [0 27]);
% set(gca,'xtick',[]);
set(gcf, 'position', [500 200 630 700]);






