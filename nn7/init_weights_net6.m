function W = init_weights_net6(net, initfn)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    antembed_bias = isfield(net, 'antembed_bias') && net.antembed_bias;
    srcembed_bias = isfield(net, 'srcembed_bias') && net.srcembed_bias;
    
    W = struct();
    W.srcembed = initfn(length(net.srcvoc) + srcembed_bias, net.srcembed);
    W.antembed = initfn(length(net.tgtvoc) + antembed_bias, net.antembed);
    W.linkAhid = initfn(net.link + 1, net.Ahid);
    W.AhidAres = initfn(net.Ahid + 1, 1);
    W.embhid = initfn(net.srcngsize * net.srcembed + net.antembed + 2, ...
        net.hidden);
    W.hidout = initfn(net.hidden + 2, net.output - 1);
end

