
z=zeros(10,2);
for i=1:50
index=find(class_update==i);
z(i,1)=i;
z(i,2)=size(index,2);
end
z