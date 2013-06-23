function targets = softmax_targets(intargets)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    tgtrowsum = sum(intargets, 2);
    emptyrows = (tgtrowsum == 0);
    tgtrowsum(emptyrows) = 1; % avoid division by zero
    targets = [bsxfun(@times, intargets, 1 ./ tgtrowsum), emptyrows];
end

