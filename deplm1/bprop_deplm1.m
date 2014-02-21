function G = bprop_deplm1(net, input, internal, output, W)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    G = struct();

    outlayer_inputgrads = output - targets;
    
    G.hidout = addbias(internal.hidden)' * outlayer_inputgrads;
    hidlayer_inputgrads = internal.hidden .* (1 - internal.hidden) .* ...
        (outlayer_inputgrads * W.hidout(2:end,:)');
    
    G.embhid = addbias(internal.embed)' * hidlayer_inputgrads;
    emblayer_inputgrads = internal.embed .* (1 - internal.embed) .* ...
        (hidlayer_inputgrads * W.embhid(2:end,:)');
    
    G.srcembed = input.srchead' * emblayer_inputgrads(:,1:net.vocembed) + ...
        input.srcdep' * emblayer_inputgrads(:,(net.vocembed + 1):(2 * net.vocembed));
    G.tgtembed = input.tgthead' * emblayer_inputgrads(:,(2 * net.vocembed + 1):(3 * net.vocembed)) + ...
        input.tgtdep' * emblayer_inputgrads(:,(3 * net.vocembed + 1):(4 * net.vocembed));

end

