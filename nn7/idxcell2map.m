function [map,outidx] = idxcell2map(inidx)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    mapidx = cellfun(@(x,i) i*ones(size(x,1),1), inidx, ...
        num2cell(1:size(inidx,2)), 'UniformOutput', false);
    map = vertcat(mapidx{:});
    range = (1:length(map))';
    mat = sparse(range, map, true);
    outidx = cellfun(@(x) range(logical(mat(:,x))), num2cell(1:length(inidx)), ...
        'UniformOutput', false);
end

