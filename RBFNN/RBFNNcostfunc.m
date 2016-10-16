function [J,grad]=RBFNNcostfunc(nn_params,X,Y,activation,nn_per_category,num_labels,lambda)
Theta=reshape(nn_params,nn_per_category*num_labels+1,num_labels);
m=size(X,1);

h=sigmoid(activation*Theta);

y_expand=zeros(size(Y,1),num_labels);
for i=1:size(Y,1)
    y_expand(i,Y(i)+1)=1;
end

 J=(-(1/m)*sum((sum((y_expand.*log(h)+((1-y_expand).*log(1-h)))))));
%J=(1/m)*sum(sum((h-y_expand).^2));
reg=(lambda/(2*m))*(((sum(sum(Theta.^2)))-sum(Theta(:,1).^2)));
J=J+reg;



delta= h-y_expand;
big_delta=delta'*activation;

temp1=Theta;
temp1(:,1)=0;

Theta_grad=(1/m)*big_delta'+(lambda*(1/m)*temp1);


% Unroll gradients
grad = Theta_grad(:);




end


