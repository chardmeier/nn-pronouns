function out = transform_weights(fn, varargin)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    
    inc = cellfun(@(x) struct2cell(orderfields(x)), varargin, 'UniformOutput', false);
    outc = cellfun(fn, inc{:}, 'UniformOutput', false);
    out = cell2struct(outc, fieldnames(orderfields(varargin{1})), 1);
end

