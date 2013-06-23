function [output, internal] = fprop_net3(net, input, weights, prediction_mode)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    W = weights_net3(net, weights);
    
    embed = zeros(input.nitems, net.srcngsize * net.srcembed + net.antembed);
    for i = 1:net.srcngsize
        embed(:,((i-1)*net.srcembed+1):(i*net.srcembed)) = W.srcembed(input.src(:,i),:);
    end
    if ~prediction_mode && isfield(net, 'dropout_src')
        dropout = randperm(input.nitems, round(net.dropout_src * input.nitems));
        embed(dropout,:) = 0;
    end
    indices = cellfun(@(v,i) i*ones(size(v)), input.ant, num2cell(1:length(input.ant))', ...
        'UniformOutput', false);
    awmat = [input.antweights{:}];
    wmat = sparse([indices{:}], 1:length(awmat), awmat);
    embed(:,(net.srcngsize * net.srcembed + 1):end) = wmat * W.antembed([input.ant{:}],:);
%     for i = 1:size(input.ant)
%         embed(i,(net.srcngsize * net.srcembed + 1):end) = input.antweights{i} * W.antembed(input.ant{i},:);
%     end
    embed = sigmoid(embed);
    
    hidden = sigmoid([ones(input.nitems, 1), input.nada, embed] * W.embhid);
    before_output = [ones(input.nitems, 1), input.nada, hidden] * W.hidout; 
 
    output = [exp(before_output), ones(input.nitems, 1)];
    output = bsxfun(@times, output, 1 ./ sum(output, 2));

    internal = struct('embed', embed, 'hidden', hidden);
end

