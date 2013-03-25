function [p,ii]=distinct(v);
p=0;
if isempty(v)
  return
end
n=length(v);
jj=find(diag(cumsum((v*ones(1,n)==ones(n,1)*v')'))==1);
p=v(jj);
k=length(jj);
ii=(v*ones(1,k)==ones(n,1)*p')*[1:k]';
