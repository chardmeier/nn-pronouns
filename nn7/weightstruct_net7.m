function W = weightstruct_net7(net, weights)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    function mat = extract(i, j)
        mat = reshape(weights(idx:(idx+i*j)), i, j);
        idx = idx + i * j;
    end

    idx = 1;
    
    W.srcembed = extract(net.srcwvec, net.srcembed);
    W.srcembjoin = extract(net.srcngsize * net.srcembed, net.srcjoin);
    W.antembed = extract(net.antwvec + 1, net.Ahid1);
    W.Ahid1Ahid2 = extract(net.Ahid1 + 1, net.Ahid2);
    W.joinhid = extract(net.srcjoin + net.Ahid2 + 2, net.hidden);
    W.hidout = extract(net.hidden + 2, net.output);
    
    if idx ~= length(weights)
        error('Extracted %d items from weight vector of length %d', idx, length(weights));
    end
end

