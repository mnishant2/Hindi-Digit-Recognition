function [J grad] = dnn3CostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_2_size, ...
                                   hidden_layer_3_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a three layer
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
Theta1 = reshape(nn_params(1 : hidden_layer_2_size*(input_layer_size+1)), ...
                 hidden_layer_2_size, (input_layer_size + 1));
            
Theta2 = reshape(nn_params((1 + (hidden_layer_2_size * (input_layer_size + 1))) : ((hidden_layer_2_size*(input_layer_size+1))+(hidden_layer_3_size * (hidden_layer_2_size + 1)))), ...
                 hidden_layer_3_size, (hidden_layer_2_size + 1));

Theta3 = reshape(nn_params((1 +((hidden_layer_2_size*(input_layer_size+1))+(hidden_layer_3_size * (hidden_layer_2_size + 1)))) : end), ...
                 num_labels, (hidden_layer_3_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

m = size(X, 1);
X = [ones(m, 1) X];
bias = ones(m,1);

% =================== FEED FORWARD PROPAGATION ======================

a1 = X;
z2 = a1*Theta1';
a2 = [bias sigmoid(z2)];
z3 = a2*Theta2';
a3 = [bias sigmoid(z3)];
z4 = a3*Theta3';
a4 = sigmoid(z4);
hypothesis=a4;

% ====================== CALCULATION OF COST ======================

y_mat = zeros(m,num_labels);
for i=1:m,
    y_mat(i,y(i)+1) = 1;
end

J = (y_mat.*log(hypothesis)) + ((1-y_mat).*log(1-hypothesis));
J = sum(sum(J));
J = (-(1*J))/m;

% ================ REGULARIZATION OF COST FUNCTION ==================

p = (Theta1.^2*(lambda))/(2*size(X,1));
q = (Theta2.^2*(lambda))/(2*size(X,1));
r = (Theta3.^2*(lambda))/(2*size(X,1));
s1 = sum(sum(p(:,2:end)));
s2 = sum(sum(q(:,2:end)));
s3 = sum(sum(r(:,2:end)));
J = J+s1+s2+s3;


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
% ====================== BACKPROPAGATION ======================
% == STEP2 ===     
delta_4 = (a4-y_mat);

% == STEP3 === 
Theta3_temp = Theta3(:,2:end);
delta_3 = (delta_4*Theta3_temp).*sigmoidGradient(z3);
Theta2_temp = Theta2(:,2:end);
delta_2 = (delta_3*Theta2_temp).*sigmoidGradient(z2);

% == STEP4 ===   
delta1_CAP = (delta_2)'*a1;
delta2_CAP = (delta_3)'*a2;
delta3_CAP = (delta_4)'*a3;

% == STEP5 === 
Theta1_grad = delta1_CAP/size(X,1);
Theta2_grad = delta2_CAP/size(X,1);
Theta3_grad = delta3_CAP/size(X,1);


% ================ REGULARIZATION OF GRADIENT =====================
p = (Theta1*(lambda))/size(X,1);
q = (Theta2*(lambda))/size(X,1);
r = (Theta3*(lambda))/size(X,1);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + p(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + q(:,2:end);
Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + r(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
