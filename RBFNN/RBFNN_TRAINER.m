%loading data
load('inputlabel5k.mat');
load('inputdata5k.mat');

%concating inputdata and inputlabel
trainingdata=[inputdata inputlabel]; 
%randomising data
a=randperm(size(inputlabel,1));
X=trainingdata(a,:);


%dividing the data into training and test set
x_train=X(1:3750,1:13);
y_train=X(1:3750,14);
x_test=X(3751:5000,1:13);
y_test=X(3751:5000,14);


nn_per_category=10; % number of neurons per category
num_labels=10;
%==========================node and beta================================

[node ,beta]= node_beta(x_train,y_train,nn_per_category,num_labels);

distance_sq=dist(x_train,node').^2;%distance b/w each input and each node
                       %operation at layer 1
        m=size(x_train,1);               

activation=exp(-distance_sq.*repmat(beta,1,m)'); % activation achieved at layer 2 

activation=[ones(m,1) activation];
%=========================================================================
initial_Theta= randInitializeWeights(nn_per_category*num_labels, num_labels);



% Unroll parameters
initial_nn_params = initial_Theta(:);


%==================backpropagation==============================
options = optimset('MaxIter',1000);

%  You should also try different values of lambda
lambda = 0.033;

% Create "short hand" for the cost function to be minimized
costFunction = @(p)RBFNNcostfunc(p,x_train,y_train,activation,nn_per_category,num_labels,lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)  
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta=reshape(nn_params,nn_per_category*num_labels+1,num_labels);
%=====================prediction========================
pred=RBFNN_predict(x_test,beta,node,Theta);

pred=pred-1;

z=(y_test-pred'==0);

k=mean(z)*100;

fprintf('Training acuracy is:%f percent',k);
