load('inputdata5k.mat');
k=50;%no of cluster
z=randperm(2000);
randsel=z(1:k);
%randsel=[1,10,20,30,40,50,60,70,80,90];
k_rand=inputdata(randsel,:);
cluster=k_rand;
cluster_update=zeros(k,13);% note 13c is number of feature involved
 threshold=3;
 rep=1000;
 
for i=1:rep
    dist_data=dist(inputdata,cluster');
%    b= cluster;
    dist_data1=compet(-1.*dist_data');
    class_update=vec2ind(dist_data1);%note:updates are transpose of what actually required
    %neural_distance=dist(cluster,cluster');
    joint_matrix=[inputdata class_update'];
    for j=1:k
       index=find(joint_matrix(:,14)==j);
      
      
        cluster_update(j,:)=mean(joint_matrix(index,1:13));
       
    end
     
        
        
      cluster=cluster_update;
end
      countsom