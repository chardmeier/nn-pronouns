function out = groupwise_softmax(in, grouping)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    mask = bsxfun(@eq, grouping, 1:max(grouping));
    vals = bsxfun(@times, in, mask);
    vals(~mask) = -inf;
    vals = bsxfun(@minus, vals, max(vals, [], 1));
    vals = exp(vals); % exp(-inf) = 0!
    vals = bsxfun(@times, vals, 1 ./ sum(vals, 1));
    out = vals(mask);
end

