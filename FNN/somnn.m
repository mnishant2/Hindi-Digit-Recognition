
load('inputdata5k.mat');
inputdata=double(inputdata);
kn=10 ;% k is no of cluster
z=randperm(2000);
yy=z(1:10);
k_rand=inputdata(yy,:);%for taking first k random initialisation for neurons
cluster=k_rand;
% for i=1:100
%     distance=dist(inputdata',cluster);
%     class_update=min(distance);
%     neural_ditance=dist(cluster',cluster);
%     cluster_update=
%     
% a=randperm(size(inputdata,1));
% X=double(inputdata(a,:));
X=double(inputdata);
x_train=X(1:3750,1:13);
x_train=x_train';
neig_thresh=max(max(dist(cluster',cluster)));
label=zeros(size(x_train,2),1);
for neig_thresh=neig_thresh:(-1*(neig_thresh))/100:0 ,
euc=negdist(cluster,x_train);
comp=compet(euc);
for i=1:10
    ind=find(comp(i,:)==1);
    m=mean(x_train(:,ind),2);
    label(ind)=i;
    d=m-cluster(i,:)';
  k = dis(i,cluster,neig_thresh);
    cluster(i,:)=m;
%     j=zeros(size(cluster,1),13);
%     j(:,13)=d';
j=numel(k);
    cluster(k,:)=cluster(k,:)+ (repmat(d,1,j))';
end
end    
    



