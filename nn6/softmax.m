function out = softmax(in, varargin)
% Compute softmax probabilities. If 'addcategory' is passed as an argument,
% an extra column of zeros is added to the inputs before computing the softmax.

    if nargin > 1 && strcmp(varargin{1}, 'addcategory')
        in2 = [in, zeros(size(in, 1), 1)];
    else
        in2 = in;
    end

    t = exp(bsxfun(@minus, in2, max(in2, [], 2)));
    out = bsxfun(@times, t, 1 ./ sum(t, 2));
end

