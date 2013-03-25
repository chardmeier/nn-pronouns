function [der,psi,nappx] = inbeder(x,p,q)
%
%             x: Input argument -- vector of length k containing values to
%                                  which beta function is integrated 
%             p,q: Input arguments -- beta shape parameters, either vectors
%                                     with same dimension as x or scalars
%             der: output -- matrix of dimension k by 6
%                            der(:,1) = I (incomplete beta function)
%                            der(:,2) = Ip
%                            der(:,3) = Ipp
%                            der(:,4) = Iq
%                            der(:,5) = Iqq
%                            der(:,6) = Ipq
%             psi: output argumens -- matrix of dimension k by 7
%                                    psi(:,1) = log[Beta(p,q)]
%                                    psi(:,2) = digamma(p)
%                                    psi(:,3) = trigamma(p)
%                                    psi(:,4) = digamma(q)
%                                    psi(:,5) = trigamma(q)
%                                    psi(:,6) = digamma(p+q)
%                                    psi(:,7) = trigamma(p+q)

%             nappx: output -- highest order approximant evaluated
%                              Interation stops if nappx > maxappx.
%                              The value of maxappx is set below.
%
err=1.e-12;
minappx=3;
maxappx=200;
n= 0
%
%          Initialize derivative vectors 
%          and check for admissability of input arguments
%

if nargin<3, 
   error('Requires three input arguments.'); 
end

[errorcode x p q] = distchck(3,x,p,q);

if errorcode > 0
   error('Requires non-scalar arguments to match in size.');
end

[k,k2]=size(x);
if k2 > 1
  error('input arguments x, p, & q must be scalars or column vectors')
end
if any((x <0)+(x > 1))
  error('Input argument x must be in [0,1] interval')
end
psi=zeros(k,7);

k1 = find(p<=0 | q<=0);
if any(k1)
   tmp = NaN;
   der(k1,:) = tmp(ones(length(k1),6)); 
end
%
%     If x >= 1 the cdf of x is 1. 
%
k2 = find(x >= 1);
if any(k2)
   der(k2,1) = ones(length(k2),1);
   der(k2,2:6)=zeros(length(k2),5);
end
k3 = find(x <= 0);
if any(k3)
   der(k3,1) = zeros(length(k3),1);
   der(k3,2:6)=zeros(length(k3),5);
end
kk = find(x > 0 & x < 1 & p > 0 & q > 0);
if any(kk)
nk=length(kk);
dr=ones(nk,6);
der=zeros(nk,6);
der_old=zeros(nk,6);
c=zeros(nk,6);
an1=zeros(nk,6);
an2=zeros(nk,6);
bn1=zeros(nk,6);
bn2=zeros(nk,6);
an1(:,1)=ones(nk,1);
an2(:,1)=ones(nk,1);
bn1(:,1)=ones(nk,1);

  
  
%
%   Compute Log Beta, digamma, and trigamma functions
%
pk=p(kk);
qk=q(kk);
psik=[betaln(pk,qk) digam(pk) trigam(pk) digam(qk) ...
	trigam(qk) digam(pk+qk) trigam(pk+qk)];
psi(kk,:)=psik;
lbet=psik(:,1);
pa=psik(:,2);
pa1=psik(:,3);
pb=psik(:,4);
pb1=psik(:,5);
pab=psik(:,6);
pab1=psik(:,7);
%
%          Use I(x,p,q) = 1- I(1-x,q,p) if x > p/(p+q)
%
xk=x(kk);
x1=xk;
omx=1-xk;
pp=pk;
qq=qk;

ii2=find(xk>pk./(pk+qk));
if any(ii2)
x1(ii2)=1-xk(ii2);
omx(ii2)=xk(ii2);
pp(ii2)=qk(ii2);
qq(ii2)=pk(ii2); 
pa(ii2)=psik(ii2,4);
pb(ii2)=psik(ii2,2);
pa1(ii2)=psik(ii2,5);
pb1(ii2)=psik(ii2,3);
end
w=x1./omx;
logx1=log(x1);
logomx=log(omx);
%
%          Compute derivatives of K(x,p,q) = x^p(1-x)^(q-1)/[p beta(p,q)]
%
c(:,1)=pp.*logx1+(qq-1).*logomx-lbet-log(pp);
c0=exp(c(:,1));
c(:,2)=logx1-1./pp-pa+pab;
c(:,3)=c(:,2).^2+1./pp.^2-pa1+pab1;
c(:,4)=logomx-pb+pab;
c(:,5)=c(:,4).^2-pb1+pab1;
c(:,6)=c(:,2).*c(:,4)+pab1;
%
%          Set counter and begin iteration
%
del=1;
while del==1
n=n+1;
%
%          Compute derivatives of an and bn with respect to p and/or q
%
[an,bn]=derconf(pp,qq,w,n);
%
%          Use forward recurrance relations to compute An, Bn,
%          and their derivatives
%
dan(:,1)=an(:,1).*an2(:,1)+bn(:,1).*an1(:,1);
dbn(:,1)=an(:,1).*bn2(:,1)+bn(:,1).*bn1(:,1);
dan(:,2)=an(:,2).*an2(:,1)+an(:,1).*an2(:,2)+bn(:,2).*an1(:,1)+ ...
    bn(:,1).*an1(:,2);
dbn(:,2)=an(:,2).*bn2(:,1)+an(:,1).*bn2(:,2)+bn(:,2).*bn1(:,1)+ ...
    bn(:,1).*bn1(:,2);
dan(:,3)=an(:,3).*an2(:,1)+2*an(:,2).*an2(:,2)+an(:,1).*an2(:,3)+ ...
    bn(:,3).*an1(:,1)+2*bn(:,2).*an1(:,2)+bn(:,1).*an1(:,3);
dbn(:,3)=an(:,3).*bn2(:,1)+2*an(:,2).*bn2(:,2)+an(:,1).*bn2(:,3)+ ...
    bn(:,3).*bn1(:,1)+2*bn(:,2).*bn1(:,2)+bn(:,1).*bn1(:,3);
dan(:,4)=an(:,4).*an2(:,1)+an(:,1).*an2(:,4)+bn(:,4).*an1(:,1)+ ...
    bn(:,1).*an1(:,4);
dbn(:,4)=an(:,4).*bn2(:,1)+an(:,1).*bn2(:,4)+bn(:,4).*bn1(:,1)+ ...
    bn(:,1).*bn1(:,4);
dan(:,5)=an(:,5).*an2(:,1)+2*an(:,4).*an2(:,4)+an(:,1).*an2(:,5)+ ...
    bn(:,5).*an1(:,1)+2*bn(:,4).*an1(:,4)+bn(:,1).*an1(:,5);
dbn(:,5)=an(:,5).*bn2(:,1)+2*an(:,4).*bn2(:,4)+an(:,1).*bn2(:,5)+ ...
    bn(:,5).*bn1(:,1)+2*bn(:,4).*bn1(:,4)+bn(:,1).*bn1(:,5);
dan(:,6)=an(:,6).*an2(:,1)+an(:,2).*an2(:,4)+an(:,4).*an2(:,2)+ ...
    an(:,1).*an2(:,6)+bn(:,6).*an1(:,1)+bn(:,2).*an1(:,4)+ ...
    bn(:,4).*an1(:,2)+bn(:,1).*an1(:,6);
dbn(:,6)=an(:,6).*bn2(:,1)+an(:,2).*bn2(:,4)+an(:,4).*bn2(:,2)+ ...
    an(:,1).*bn2(:,6)+bn(:,6).*bn1(:,1)+bn(:,2).*bn1(:,4)+ ...
    bn(:,4).*bn1(:,2)+bn(:,1).*bn1(:,6);
%
%          Scale derivatives to prevent overflow
%
Rn=dan(:,1);
iii1=[1:nk]';
iii2=find(abs(dbn(:,1))>abs(dan(:,1)));
iii1(iii2)=[];
Rn(iii2)=dbn(iii2,1);
an1=an1./(Rn*ones(1,6));  
bn1=bn1./(Rn*ones(1,6));
dan(:,[2:6]')=dan(:,[2:6]')./(Rn*ones(1,5));  
dbn(:,[2:6]')=dbn(:,[2:6]')./(Rn*ones(1,5));
dbn(iii1,1)=dbn(iii1,1)./dan(iii1,1);
dan(iii1,1)=ones(size(iii1));
dan(iii2,1)=dan(iii2,1)./dbn(iii2,1);
dbn(iii2,1)=ones(size(iii2));

%
%          Compute components of derivatives of the nth approximant
%
dr(:,1)=dan(:,1)./dbn(:,1);
Rn = dr(:,1);
dr(:,2) = (dan(:,2)-Rn.*dbn(:,2))./dbn(:,1);
dr(:,3) = (-2*dan(:,2).*dbn(:,2)+2*Rn.*dbn(:,2).^2)./dbn(:,1).^2+ ...
    (dan(:,3)-Rn.*dbn(:,3))./dbn(:,1);
dr(:,4) = (dan(:,4)-Rn.*dbn(:,4))./dbn(:,1);
dr(:,5) = (-2*dan(:,4).*dbn(:,4)+2*Rn.*dbn(:,4).^2)./dbn(:,1).^2+ ...
    (dan(:,5)-Rn.*dbn(:,5))./dbn(:,1);
dr(:,6) = (-dan(:,2).*dbn(:,4)-dan(:,4).*dbn(:,2)+ ...
    2*Rn.*dbn(:,2).*dbn(:,4))./dbn(:,1).^2+(dan(:,6)-Rn.*dbn(:,6))./dbn(:,1);
%
%          Save terms corresponding to approximants n-1 and n-2
%
an2=an1;
an1=dan;
bn2=bn1;
bn1=dbn;
%
%          Compute nth approximants
%
pr=zeros(nk,1);
iii=find(dr(:,1)>0);
pr(iii)=exp(c(iii,1)+log(dr(iii,1)));
der(:,1)=pr;
der(:,2)=pr.*c(:,2)+c0.*dr(:,2);
der(:,3)=pr.*c(:,3)+2*c0.*c(:,2).*dr(:,2)+c0.*dr(:,3);
der(:,4)=pr.*c(:,4)+c0.*dr(:,4);
der(:,5)=pr.*c(:,5)+2.*c0.*c(:,4).*dr(:,4)+c0.*dr(:,5);
der(:,6)=pr.*c(:,6)+c0.*c(:,4).*dr(:,2)+c0.*c(:,2).*dr(:,4)+c0.*dr(:,6);
%
%          Check for convergence, check for maximum and minimum iterations.
%          
d1=max(err*ones(nk,6),abs(der));
d1=abs(der_old-der)./d1;
d=max(max(d1));
der_old=der;
if n < minappx
  d=1;
end
if n >= maxappx
d=0;
end
del=0;
if d > err
 del=1;
end
end

%
%          Adjust results if I(x,p,q) = 1- I(1-x,q,p) was used
%
der(ii2,1)=1-der(ii2,1);
c0=der(ii2,2);
der(ii2,2)=-der(ii2,4);
der(ii2,4)=-c0;
c0=der(ii2,3);
der(ii2,3)=-der(ii2,5);
der(ii2,5)=-c0;
der(ii2,6)=-der(ii2,6);
end
nappx=n;
