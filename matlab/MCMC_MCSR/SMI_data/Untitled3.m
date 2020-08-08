a=1:1:100;
b=1000*power(0.9,a);
plot(b,a);



















%     inversion_variance=load('.\SeismicData\testdata\varaverage52_2001test.txt');    %%load varance
%     initial_impedence=load('.\SeismicData\testdata\modelaveragr52_2001test.txt');   %%load inital impendence
%     initial_impedence=initial_impedence(1:2001,110*98+1:110*98+100);
%     inversion_variance=inversion_variance(1:2001,110*98+1:110*98+100);
%     dicSavePath1 = sprintf('./SeismicData/inline99_initalmodel_52.mat');
%     dicSavePath2 = sprintf('./SeismicData/inline99_varance_52.mat');
%     save(dicSavePath1, 'initial_impedence');
%     save(dicSavePath2, 'inversion_variance');
    
    