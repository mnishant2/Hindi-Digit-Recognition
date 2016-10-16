function k= dis(i,cluster,neig_thresh)
% for i=1:100
%     distance=dist(inputdata',cluster);
%     class_update=min(distance);
    neural_distance=dist(cluster(i,:),cluster(i,:)');
    k=find(dist(cluster(i,:),cluster(i,:)')<=neig_thresh);
end
    