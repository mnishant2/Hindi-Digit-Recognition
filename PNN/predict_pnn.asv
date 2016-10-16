function p = predict_pnn(Q, T, X,spread)
m=size(Q,1);
n=size(X,2);
bias=zeros(m,n)+sqrt(-log(.5))/spread;
euc=dist(Q,X);
target=(euc).*(bias);
target=radbas(target);
res=T*target;
comp=compet(res);
p=vec2ind(comp);
end