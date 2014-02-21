function W = init_weights_deplm1(net, initfn)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
 
    nweights = 2 * (net.vocsize + 1) * net.vocembed + ...
        (net.nonvoc + 4 * net.vocembed + 1) * net.hidden + ...
        (net.hidden + 1) * net.output;
    
    W = initfn(nweights, 1);
end

