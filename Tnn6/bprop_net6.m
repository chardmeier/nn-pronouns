function G = bprop_net6(net, input, internal, output, W)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    antembed_bias = isfield(net, 'antembed_bias') && net.antembed_bias;
    srcembed_bias = isfield(net, 'srcembed_bias') && net.srcembed_bias;
    
    G = struct();

    outlayer_inputgrads = output - input.targets;
    
    G.hidout = addbias([input.nada, internal.hidden])' * outlayer_inputgrads;
    hidlayer_inputgrads = internal.hidden .* (1 - internal.hidden) .* ...
        (outlayer_inputgrads * W.hidout(3:end,:)');
    
    G.embhid = addbias([input.nada, internal.embed])' * hidlayer_inputgrads;
    emblayer_outputgrads = hidlayer_inputgrads * W.embhid(3:end,:)';
    
    srcembed = reshape(internal.embed(:,1:(net.srcngsize*net.srcembed)), ...
        input.nitems, net.srcembed, net.srcngsize);
    srcemb_inputgrads = srcembed .* (1 - srcembed) .* ...
        reshape(emblayer_outputgrads(:,1:(net.srcngsize*net.srcembed)), ...
        input.nitems, net.srcembed, net.srcngsize);

    srcvocsize = length(net.srcvoc);
    G.srcembed = sparse(srcvocsize + srcembed_bias, net.srcembed);
    for i = 1:net.srcngsize
        srcin = input.src(:,((i-1)*srcvocsize+1):(i*srcvocsize));
        if srcembed_bias
            src = addbias(srcin);
        else
            src = srcin;
        end
        G.srcembed = G.srcembed + ...
            src' * sparse(reshape(srcemb_inputgrads(:,:,i), input.nitems, net.srcembed));
    end
    
    wantfeatures_grads = emblayer_outputgrads(:,(net.srcngsize*net.srcembed+1):end);

    antfeatures_inputgrads = internal.antfeatures .* (1 - internal.antfeatures) .* ...
        (internal.wAres' * wantfeatures_grads);
    if antembed_bias
        ant = addbias(input.ant);
    else
        ant = input.ant;
    end
    G.antembed = ant' * sparse(antfeatures_inputgrads);
    
    % Ares logistic layer
%     Ares_inputgrads = internal.Ares .* (1 - internal.Ares) .* ...
%         sum(internal.antfeatures .* wantfeatures_grads(input.antmap,:), 2);
%     G.AhidAres = addbias(internal.Ahid)' * Ares_inputgrads;
% 
%     Ahid_inputgrads = internal.Ahid .* (1 - internal.Ahid) .* ...
%         (Ares_inputgrads * W.AhidAres(2:end,:)');
%     G.linkAhid = addbias(input.link)' * Ahid_inputgrads;

    % Ares softmax layer
    Ares_outputgrads = sum(internal.antfeatures .* wantfeatures_grads(input.antmap,:), 2);
    Ares_wsuminput_tmp = bsxfun(@times, internal.Ahid, internal.Ares);
    Ares_wsuminput_c = cellfun(@(x) repmat(sum(Ares_wsuminput_tmp(x,:), 1), size(x, 1), 1), ...
        input.antidx, 'UniformOutput', false);
    Ares_wsuminput = vertcat(Ares_wsuminput_c{:});
    G.AhidAres = [0; ...
        sum(bsxfun(@times, internal.Ahid - Ares_wsuminput, internal.Ares .* Ares_outputgrads), 1)'];
    
    dAresin_dAhidin = bsxfun(@times, internal.Ahid .* (1 - internal.Ahid), W.AhidAres(2:end,:)');
    dAresin_dlinkAhid = bsxfun(@times, reshape(dAresin_dAhidin, [], 1, net.Ahid), ...
        addbias(full(input.link)));
    dAresin_dlinkAhid_wsum_tmp = bsxfun(@times, dAresin_dlinkAhid, internal.Ares);
    dAresin_dlinkAhid_wsum_c = cellfun(@(x) repmat(sum(dAresin_dlinkAhid_wsum_tmp(x,:,:), 1), [size(x, 1) 1 1]), ...
        input.antidx, 'UniformOutput', false);
    dAresin_dlinkAhid_wsum = cat(1, dAresin_dlinkAhid_wsum_c{:});
    G.linkAhid = reshape(sum(bsxfun(@times, dAresin_dlinkAhid - dAresin_dlinkAhid_wsum, ...
        internal.Ares .* Ares_outputgrads), 1), ...
        size(W.linkAhid));
    
    %Ares_inputgrads = internal.Ares .* (1 - internal.Ares) .* Ares_outputgrads;
    %Ahid_inputgrads = internal.Ahid .* (1 - internal.Ahid) .* (Ares_inputgrads * W.AhidAres(2:end,:)');
    %G.linkAhid = addbias(input.link)' * Ahid_inputgrads;
end

