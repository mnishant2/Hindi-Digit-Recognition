function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   hidden_layer3_size, ...
                                   num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):(0 + (hidden_layer1_size * (input_layer_size + 1)))+hidden_layer2_size*(hidden_layer1_size+1)), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));
             
Theta3 = reshape(nn_params((1 + (hidden_layer2_size * (hidden_layer1_size + 1))):(0 + (hidden_layer2_size * (hidden_layer1_size + 1)))+hidden_layer3_size*(hidden_layer2_size+1)), ...
                 hidden_layer3_size, (hidden_layer2_size + 1));
             
Theta4 =  reshape(nn_params((1 + (hidden_layer3_size * (hidden_layer2_size + 1))):(0 + (hidden_layer3_size * (hidden_layer2_size + 1)))+num_labels*(hidden_layer3_size+1)), ...
                 num_labels, (hidden_layer3_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%===========================common variaable declaration===================

bias=ones(m,1);     % for adding 1 in all features
z_1=X;
a_1=[bias z_1];


z_2=a_1*Theta1';
a_2=[bias sigmoid(z_2)];% layer 2

z_3=a_2*Theta2';
a_3=[bias sigmoid(z_3)];% layer 3


z_4=a_3*Theta3';
a_4=[bias sigmoid(z_4)];% layer4


z_5=a_4*Theta4';
a_5= sigmoid(z_5);       %output layer    


%=========%cost_function====================================================


h=a_5;

y_expand=zeros(size(y,1),num_labels);
for i=1:size(y,1)
    y_expand(i,y(i))=1;
end
J=(-(1/m)*sum((sum((y_expand.*log(h)+((1-y_expand).*log(1-h)))))));
%========================regularisation+======================

%reg=(lambda/(2*m))*(((sum(sum(Theta1.^2)))-sum(Theta1(:,1).^2))+((sum(sum(Theta3.^2)))-sum(Theta3(:,1).^2))+((sum(sum(Theta2.^2))))...
  %  -sum(Theta2(:,1).^2)+((sum(sum(Theta4.^2)))-sum(Theta4(:,1).^2)));
%J=J+reg;





%==============backpropagation====================================================





delta_5= a_5-y_expand;
delta_4=delta_5*(Theta4(:,2:end)).*sigmoidGradient(z_4);
delta_3=delta_4*(Theta3(:,2:end)).*sigmoidGradient(z_3);
delta_2=delta_3*(Theta2(:,2:end)).*sigmoidGradient(z_2);


big_delta_4=delta_5'*a_4;
big_delta_3=delta_4'*a_3;
big_delta_2=delta_3'*a_2;
big_delta_1=delta_2'*a_1;


temp1=Theta1;
temp1(:,1)=0;

temp2=Theta2;
temp2(:,1)=0;

                    %TEMP is used do as to eliminate the ias term 1
                    %while calcuating with regularization
temp3=Theta3;
temp3(:,1)=0;

temp4=Theta4;
temp4(:,1)=0;



Theta1_grad=(1/m)*big_delta_1+(lambda*(1/m)*temp1);
Theta2_grad=(1/m)*big_delta_2+(lambda*(1/m)*temp2);
Theta3_grad=(1/m)*big_delta_3+(lambda*(1/m)*temp3);
Theta4_grad=(1/m)*big_delta_4+(lambda*(1/m)*temp4);








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:); Theta4_grad(:)];


end
