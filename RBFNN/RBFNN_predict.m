function p =RBFNN_predict(X,beta,node,Theta)
distance_sq=dist(X,node').^2;%distance b/w each input and each node
                       %operation at layer 1
        m=size(X,1);               

activation=exp(-distance_sq.*repmat(beta,1,m)'); % activation achieved at layer 2 

activation=[ones(m,1) activation];
%==================================
h=sigmoid(activation*Theta);
[~,p]=max(h');
end