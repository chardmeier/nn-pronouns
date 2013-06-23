function W = weights_net3(net, weights)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    widx = 1;
    nweights = length(net.srcvoc) * net.srcembed;
    Wsrcembed = reshape(weights(widx:(widx+nweights-1)), length(net.srcvoc), net.srcembed);
    
    widx = widx + nweights;
    nweights = length(net.tgtvoc) * net.antembed;
    Wantembed = reshape(weights(widx:(widx+nweights-1)), length(net.tgtvoc), net.antembed);
    
    widx = widx + nweights;
    nweights = (net.srcngsize * net.srcembed + net.antembed + 2) * net.hidden;
    Wembhid = reshape(weights(widx:(widx+nweights-1)), net.srcngsize * net.srcembed + net.antembed + 2, net.hidden);
    
    widx = widx + nweights;
    nweights = (net.hidden + 2) * net.output;
    Whidout = reshape(weights(widx:(widx+nweights-1)), net.hidden + 2, net.output);
    
    W = struct('srcembed', Wsrcembed, 'antembed', Wantembed, 'embhid', Wembhid, 'hidout', Whidout);
end

