load('inputdata5k.mat');
k=50;%no of cluster
z=randperm(2000);
randsel=z(1:k);
data=zeros(500,13);
for tty=0:100:4900
   i=tty/10;
    data(i+1:i+10,:)=inputdata(tty+1:tty+10,:);
end

%randsel=[1,10,20,30,40,50,60,70,80,90];
% k_rand=inputdata(randsel,:);
inputdata=data;
cluster=rand(50,13);
cluster_update=zeros(k,13);% note 13c is number of feature involved
 threshold=3;
 rep=100;
 beta=0.33;
for i=1:rep
    dist_data=dist(inputdata,cluster');
   b= cluster;
    dist_data1=compet(-1.*dist_data');
    class_update=vec2ind(dist_data1);%note:updates are transpose of what actually required
    neural_distance=dist(cluster,cluster');
    joint_matrix=[inputdata class_update'];
    for j=1:k
       index=find(joint_matrix(:,14)==j);
      
      
        cluster_update(j,:)=mean(joint_matrix(index,1:13));
       
    end
       difference=(cluster_update-cluster);
        threshold=3;
        beta=beta/(1+rep/100);
      cluster=cluster_update;%==========================
       ref=(neural_distance<threshold).*(neural_distance~=0);
       for f=1:k
          alpha= find(ref(f,:)==1);
    
          cluster(alpha,:)=cluster(alpha,:)+beta.*repmat(difference(f,:)',1,size(alpha,2))';
          
       end
    
    
end
countsom
