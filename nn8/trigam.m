function p = trigam(Z)
%
%            Trigamma Function
%
[k1,k2]=size(Z);
z=vec(Z);
if any(z<=0)
  error('Psii requires positive arguments')
end
k=k1*k2;
j=max(0,ceil(13-min(z)));
z1=z+j;
p=1./(z1)+1./(2*z1.^2)+1./(6*z1.^3)-1./(30*z1.^5) ...
    +1./(42*z1.^7)-1./(30*z1.^9) +1./(66*z1.^11);
if j>0
   i=[0:j-1]';
   if j==1
      p=p+1./z.^2;
     else
      p=p+sum([1./((z*ones(1,j)+ones(k,1)*i').^2)]')';
   end
end
p=devec(p,k1,k2);


