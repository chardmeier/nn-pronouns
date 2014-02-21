function [output, internal] = fprop_deplm1(net, input, W, prediction_mode)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    if isfield(net, 'dropout')
        if ~prediction_mode
            dropout = @(x) x .* (rand(size(x)) > net.dropout);
        else
            dropout = @(x) (1 - net.dropout) * x;
        end
    else
        dropout = @(x) x;
    end
    
    embed = zeros(input.nitems, 4 * net.vocembed + net.nonvoc + 1);
    embed(:,1) = 1;
    embed(:,2:(net.vocembed + 1)) = dropout(sigmoid(input.srchead * W.srcembed));
    embed(:,(net.vocembed + 2):(2 * net.vocembed + 1)) = dropout(sigmoid(input.srcdep * W.srcembed));
    embed(:,(2 * net.vocembed + 2):(3 * net.vocembed + 1)) = dropout(sigmoid(input.tgthead * W.tgtembed));
    embed(:,(3 * net.vocembed + 2):(4 * net.vocembed + 1)) = dropout(sigmoid(input.tgtdep * W.tgtembed));
    embed(:,(4 * net.vocembed + 2):end) = input.nonvoc;
    
    hidden = dropout(sigmoid(embed * W.embhid));
    output = sigmoid(hidden * W.hidout); 

    internal = struct('embed', embed, 'hidden', hidden);
end

