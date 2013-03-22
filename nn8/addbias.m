function out = addbias(in)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    if isa(in, 'gpuArray')
        out = [gpuArray.ones(size(in, 1), 1), in];
    else
        out = [ones(size(in, 1), 1), in];
    end
end

