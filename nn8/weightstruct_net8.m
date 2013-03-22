function W = weightstruct_net8(net, weights)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    function mat = extract(i, j)
        mat = reshape(weights(idx:(idx+i*j-1)), i, j);
        idx = idx + i * j;
    end

    idx = 1;
    
    W.srcembed = extract(net.srcwvec, net.srcembed);
    W.srcembjoin = extract(net.srcprons + net.srcngsize * net.srcembed + 1, net.srcjoin);
    W.linkLhid = extract(net.link + 1, net.Lhid);
    W.LhidLres = extract(net.Lhid + 1, 1);
    W.antembed = extract(net.antwvec + 1, net.Ahid1);
    W.Ahid1Ahid2 = extract(net.Ahid1 + 1, net.Ahid2);
    W.joinhid = extract(net.srcjoin + net.Ahid2 + 2, net.hidden);
    W.hidout = extract(net.hidden + 2, net.output - 1);
    
    if idx ~= length(weights) + 1
        error('Extracted %d items from weight vector of length %d', idx, length(weights));
    end
end

