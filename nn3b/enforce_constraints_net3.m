function outweights = enforce_constraints_net3(net, weights)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    W = weights_net3(net, weights);
    W.embhid = W.embhid * diag(min(1, net.weightconstraint ./ sum(W.embhid .^ 2)));
    W.hidout = W.hidout * diag(min(1, net.weightconstraint ./ sum(W.hidout .^ 2)));
    outweights = [W.srcembed(:); W.antembed(:); W.embhid(:); W.hidout(:)]';
end

