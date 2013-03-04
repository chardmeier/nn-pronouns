function G = bprop_net7(net, input, internal, output, W)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    G = struct();

    % The last output of the softmax is unconnected
    output = output(:,1:end-1);
    targets = input.targets(:,1:end-1);

    outlayer_inputgrads = output - targets;
    
    G.hidout = addbias([input.nada, internal.hidden])' * outlayer_inputgrads;
    hidlayer_inputgrads = net.transfer.hidden.df(internal.hidden) .* ...
        (outlayer_inputgrads * W.hidout(3:end,:)');
    
    G.joinhid = addbias([input.nada, internal.join])' * hidlayer_inputgrads;
    joinlayer_outputgrads = hidlayer_inputgrads * W.embhid(3:end,:)';
    
    srcjoin_inputgrads = net.transfer.srcjoin.df(joinlayer_outputgrads(:,1:net.srcjoin));
    G.srcembjoin = reshape(internal.srcembed, input.nitems, net.srcembed * net.srcngsize)' * srcjoin_inputgrads;
    srcemblayer_outputgrads = srcjoin_inputgrads * W.srcembjoin(2:end,:)';
    
    srcemb_inputgrads = net.transfer.srcembed.df(internal.srcembed) .* ...
        reshape(srcemblayer_outputgrads, ...
        input.nitems, net.srcembed, net.srcngsize);

    srcvocsize = length(net.srcvoc);
    G.srcembed = sparse(srcvocsize + net.srcembed_bias, net.srcembed);
    for i = 1:net.srcngsize
        srcin = input.srcwvecs(input.src(:,i),:);
        if net.srcembed_bias
            src = addbias(srcin);
        else
            src = srcin;
        end
        G.srcembed = G.srcembed + ...
            src' * sparse(reshape(srcemb_inputgrads(:,:,i), input.nitems, net.srcembed));
    end
    
    wantfeatures_grads = joinlayer_outputgrads(:,(net.srcjoin+1):end);

    Ahid2_inputgrads = net.transfer.Ahid2.df(internal.Ahid2) .* ...
        (internal.wLres' * wantfeatures_grads);
    G.Ahid1Ahid2 = addbias(internal.Ahid1)' * Ahid2_inputgrads;
    
    Ahid1_inputgrads = net.transfer.Ahid1.df(internal.Ahid1) .* ...
        (Ahid2_inputgrads * W.Ahid1Ahid2(2:end,:)');
    G.antembed = addbias(input.ant)' * sparse(Ahid1_inputgrads);
    
    % Ares logistic layer
%     Ares_inputgrads = internal.Ares .* (1 - internal.Ares) .* ...
%         sum(internal.antfeatures .* wantfeatures_grads(input.antmap,:), 2);
%     G.AhidAres = addbias(internal.Ahid)' * Ares_inputgrads;
% 
%     Ahid_inputgrads = internal.Ahid .* (1 - internal.Ahid) .* ...
%         (Ares_inputgrads * W.AhidAres(2:end,:)');
%     G.linkAhid = addbias(input.link)' * Ahid_inputgrads;

    % Ares softmax layer
    Lres_outputgrads = sum(internal.antfeatures .* wantfeatures_grads(input.antmap,:), 2);
    Lres_wsuminput_tmp = bsxfun(@times, internal.Lhid, internal.Lres);
    Lres_wsuminput_c = cellfun(@(x) repmat(sum(Lres_wsuminput_tmp(x,:), 1), size(x, 1), 1), ...
        input.antidx, 'UniformOutput', false);
    Lres_wsuminput = vertcat(Lres_wsuminput_c{:});
    G.LhidLres = [0; ...
        sum(bsxfun(@times, internal.Lhid - Lres_wsuminput, internal.Lres .* Lres_outputgrads), 1)'];
    
    dLresin_dLhidin = bsxfun(@times, net.transfer.Lhid.df(internal.Lhid), W.LhidLres(2:end,:)');
    dLresin_dlinkLhid = bsxfun(@times, reshape(dLresin_dLhidin, [], 1, net.Lhid), ...
        addbias(full(input.link)));
    dLresin_dlinkLhid_wsum_tmp = bsxfun(@times, dLresin_dlinkLhid, internal.Lres);
    dLresin_dlinkLhid_wsum_c = cellfun(@(x) repmat(sum(dLresin_dlinkLhid_wsum_tmp(x,:,:), 1), [size(x, 1) 1 1]), ...
        input.antidx, 'UniformOutput', false);
    dLresin_dlinkLhid_wsum = cat(1, dLresin_dlinkLhid_wsum_c{:});
    G.linkLhid = reshape(sum(bsxfun(@times, dLresin_dlinkLhid - dLresin_dlinkLhid_wsum, ...
        internal.Lres .* Lres_outputgrads), 1), size(W.linkLhid));
    
    %Ares_inputgrads = internal.Lres .* (1 - internal.Lres) .* Lres_outputgrads;
    %Ahid_inputgrads = internal.Lhid .* (1 - internal.Lhid) .* (Lres_inputgrads * W.LhidLres(2:end,:)');
    %G.linkLhid = addbias(input.link)' * Lhid_inputgrads;
    
    if isfield(net, 'regulariser') && net.regulariser > 0
        G = transform_weights(@(x,y) x + net.regulariser * y, G, W);
    end

    if isfield(net, 'l1regulariser') && net.l1regulariser > 0
        G = transform_weights(@(x,y) x + net.l1regulariser * sign(y), G, W);
    end
end

