function out = addbias(in)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    out = [ones(size(in, 1), 1), in];
end

