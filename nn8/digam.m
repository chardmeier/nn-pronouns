function p = digam(Z)
%
%           Digamma Function
%
[k1,k2]=size(Z);
z=vec(Z);
if any(z<=0)
  error('Psi requires positive arguments')
end
k=k1*k2;
j=max(0,ceil(13-min(z)));
z1=z+j;
p=log(z1)-1./(2*z1)-1./(12*z1.^2)+1./(120*z1.^4)-1./(252*z1.^6) ...
    +1./(240*z1.^8)-1./(132*z1.^10);
if j>0
  i=[0:j-1]';
  if j==1
    p=p-1./z;
   else
    p=p-sum([1./(z*ones(1,j)+ones(k,1)*i')]')';
  end
end
p=devec(p,k1,k2);
