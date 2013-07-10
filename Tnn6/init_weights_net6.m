function W = init_weights_net6(net, initfn)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    
    antembed_bias = isfield(net, 'antembed_bias') && net.antembed_bias;
    srcembed_bias = isfield(net, 'srcembed_bias') && net.srcembed_bias;
  
    
    nweights = (length(net.srcvoc) + srcembed_bias) * net.srcembed + ...
        (length(net.tgtvoc) + antembed_bias) * net.antembed + ...
        (net.link + 1) * net.Ahid + ...
        net.Ahid + 1 + ...
        (net.srcngsize * net.srcembed + net.antembed + 2) * net.hidden + ...
        (net.hidden + 2) * (net.output);
    
    W = initfn(nweights, 1);
end

