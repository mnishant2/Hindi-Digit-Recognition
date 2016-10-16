% this is the main file 

% Load Training Data
fprintf('Loading Training DATA...\n');

load('label0db5k.mat');

load('data0db5k.mat');
trainingdata=[inputdata inputlabel];


fprintf('Applying backpropagation...');

a=randperm(size(trainingdata,1));
X=double(trainingdata(a,:));

x_train=X(1:3750,1:13);
y_train=X(1:3750,14);
x_test=X(3751:5000,1:13);
y_test=X(3751:5000,14);


y_train=double(y_train);
input_layer_size=size(x_train,2);
hidden_layer_2_size=128;
hidden_layer_3_size=64;
hidden_layer_4_size=32;
num_labels=10;

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_2_size);
initial_Theta2 = randInitializeWeights(hidden_layer_2_size,hidden_layer_3_size);
initial_Theta3 = randInitializeWeights(hidden_layer_3_size, hidden_layer_4_size);
initial_Theta4 = randInitializeWeights(hidden_layer_4_size, num_labels);


% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:);initial_Theta3(:);initial_Theta4(:)];
%initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:);Theta4(:)];


%==================backpropagation==============================
options = optimset('GradObj', 'on', 'MaxIter', 5000);
%options = optimset ('MaxIter', 500);

%  You should also try different values of lambda
lambda = 0.1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p)dnn4CostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_2_size, ...
                                   hidden_layer_3_size, ...
                                   hidden_layer_4_size, ...
                                   num_labels, x_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)  
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1 : hidden_layer_2_size*(input_layer_size+1)), ...
                 hidden_layer_2_size, (input_layer_size + 1));
            
Theta2 = reshape(nn_params((1 + (hidden_layer_2_size * (input_layer_size + 1))) : ((hidden_layer_2_size*(input_layer_size+1))+(hidden_layer_3_size * (hidden_layer_2_size + 1)))), ...
                 hidden_layer_3_size, (hidden_layer_2_size + 1));
             
Theta3 = reshape(nn_params((1 + ((hidden_layer_2_size*(input_layer_size+1))+(hidden_layer_3_size * (hidden_layer_2_size + 1)))) : ((hidden_layer_2_size*(input_layer_size+1))+(hidden_layer_3_size * (hidden_layer_2_size + 1)) + (hidden_layer_4_size * (hidden_layer_3_size + 1)))), ...
                 hidden_layer_4_size, (hidden_layer_3_size + 1));

Theta4 = reshape(nn_params((1 +(hidden_layer_2_size*(input_layer_size+1))+(hidden_layer_3_size * (hidden_layer_2_size + 1)) + (hidden_layer_4_size * (hidden_layer_3_size + 1))) : end), ...
                 num_labels, (hidden_layer_4_size + 1));

save('neural_param.mat','Theta1','Theta2','Theta3','Theta4');
%fprintf('end of training\n');
pred=dnn4predict(Theta1,Theta2,Theta3,Theta4,x_test);

pred=pred-1;
z=(y_test-pred==0);

k=mean(z)*100;

fprintf('Training acuracy is:%f percent',k);