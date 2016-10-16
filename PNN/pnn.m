fprintf('Loading Training DATA...\n');

load('inputlabel5k.mat');
load('inputdata5k.mat');
trainingdata=[inputdata inputlabel];
a=randperm(size(trainingdata,1));
X=double(trainingdata(a,:));
x_train=X(1:3750,1:13);
y_train=X(1:3750,14);
% x_test=X(751:1000,1:13);
y_test=X(751:1000,14);
x_test=X(751:1000,1:13);
spread=0.33;
y_train=double(y_train);
input_layer_size=size(x_train,2);
Q=size(x_train,1);
input_weights=x_train;
% input_bias=zeros(Q,1)+sqrt(-log(.5))/spread;
target_matrix=ind2vec((y_train+1)');
pred=predict_pnn(x_train,target_matrix,x_test,spread);
% pred=pred-1;
% z=(y_test-pred'==0);
% k=mean(z)*100;
% sprintf('Training acuracy is:%f percent',k)




