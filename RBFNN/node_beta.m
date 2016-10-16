function [node ,beta]= node_beta(X,Y,nn_per_category,num_labels)
    node=zeros(num_labels*nn_per_category,size(X,2));
    
    sigma=zeros(num_labels*nn_per_category,1);
    f=0;
for i=1:num_labels
    X_cat=X((Y==i-1),:);
    len=size(X_cat,1);
    div=floor(len/nn_per_category);
    n=1;
    for j=1:nn_per_category
        
        temp=mean(X_cat(n:n+div-1,:),1);
        
        f=f+1;
        node(f,:)=temp;
        sigma(f)=mean(dist(temp,X_cat'));
        n=n+div;
    end
end

beta=(1/2)*sigma.^(-2);  

end
        