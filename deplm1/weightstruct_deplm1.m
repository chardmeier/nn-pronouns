function W = weightstruct_deplm1(net, weights)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    function mat = extract(i, j)
        mat = reshape(weights(idx:(idx+i*j-1)), i, j);
        idx = idx + i * j;
    end

    W.srcembed = extract(net.vocsize + 1, net.vocembed);
    W.tgtembed = extract(net.vocsize + 1, net.vocembed);
    W.embhid = extract(net.nonvoc + 4 * net.vocembed + 1, net.hidden);
    W.hidout = extract(net.hidden + 1, net.output);
    
    if idx ~= length(weights) + 1
        error('Extracted %d items from weight vector of length %d', idx, length(weights));
    end
end