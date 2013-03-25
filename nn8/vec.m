function x=vec(T);
[a,b]=size(T);
x=zeros(a*b,1);
x(:)=T;
