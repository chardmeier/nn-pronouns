function W = init_weights_net6(net, initfn)
% Initialise NN6 network weights.
    
    antembed_bias = isfield(net, 'antembed_bias') && net.antembed_bias;
    srcembed_bias = isfield(net, 'srcembed_bias') && net.srcembed_bias;
  
    
    nweights = (length(net.srcvoc) + srcembed_bias) * net.srcembed + ...
        (length(net.tgtvoc) + antembed_bias) * net.antembed + ...
        (net.link + 1) * net.Ahid + ...
        net.Ahid + 1 + ...
        (net.srcngsize * net.srcembed + net.antembed + 2) * net.hidden + ...
        (net.hidden + 2) * (net.output - 1);
    
    W = initfn(nweights, 1);
end

