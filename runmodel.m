clear
clc
close all

[trainset,trainlabel,valset,vallabel,testset,testlabel]=get_data();

bpmodel=model([13,6,3],0.0,false);
bpmodel.train(trainset,trainlabel',5e-4,1e-3,10000,145,valset,vallabel',50);
save('bomodel.mat','bpmodel')

load('bomodel.mat', 'bpmodel')
[test_pred,test_acc]=test(bpmodel,testset,testlabel')
