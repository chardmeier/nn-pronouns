function bin = rito01(ri)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    tiny = 1e-30;
    nri = bsxfun(@times, ri, 1 ./ (sqrt(sum(ri .^ 2, 2)) + tiny));
    bin = (nri + 1) / 2;
end

