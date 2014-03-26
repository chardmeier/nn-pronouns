function out = softmax(in, varargin)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    if nargin > 1 && strcmp(varargin{1}, 'addcategory')
        in2 = [in, zeros(size(in, 1), 1)];
    else
        in2 = in;
    end

    t = exp(bsxfun(@minus, in2, max(in2, [], 2)));
    out = bsxfun(@times, t, 1 ./ sum(t, 2));
end

