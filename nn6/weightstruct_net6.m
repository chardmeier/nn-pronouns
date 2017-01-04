function W = weightstruct_net6(net, weights)
% Convert NN6 weights from flat vector to structure.

    function mat = extract(i, j)
        mat = reshape(weights(idx:(idx+i*j-1)), i, j);
        idx = idx + i * j;
    end

    antembed_bias = isfield(net, 'antembed_bias') && net.antembed_bias;
    srcembed_bias = isfield(net, 'srcembed_bias') && net.srcembed_bias;
    
    idx = 1;
    
    W.srcembed = extract(length(net.srcvoc) + srcembed_bias, net.srcembed);
    W.antembed = extract(length(net.tgtvoc) + antembed_bias, net.antembed);
    W.linkAhid = extract(net.link + 1, net.Ahid);
    W.AhidAres = extract(net.Ahid + 1, 1);
    W.embhid = extract(net.srcngsize * net.srcembed + net.antembed + 2, net.hidden);
    W.hidout = extract(net.hidden + 2, net.output - 1);
    
    if idx ~= length(weights) + 1
        error('Extracted %d items from weight vector of length %d', idx, length(weights));
    end
end
