function [an,bn]=derconf(p,q,w,n)
%
%     Compute derivatives of an and bn with respect to p and/or q
%
k=length(w);
an=zeros(k,6);
bn=zeros(k,6);
F = w.*q./p;
if n==1 
t1=1-1./(p+1);
t2=1-1./q;
t3=1-2./(p+2);
t4=1.-2./q;
an(:,1)=t1.*t2.*F;
an(:,2)=-an(:,1)./(p+1);
an(:,3)=-2*an(:,2)./(p+1);
an(:,4)=t1.*F./q;
an(:,5)=zeros(k,1);
an(:,6)=-an(:,4)./(p+1);
bn(:,1)=1-t3.*t4.*F;
bn(:,2)=t3.*t4.*F./(p+2);
bn(:,3)=-2*bn(:,2)./(p+2);
bn(:,4)=-t3.*F./q;
bn(:,6)=-bn(:,4)./(p+2);
else
[an,bn]=subd(n,p,q,F);
end
