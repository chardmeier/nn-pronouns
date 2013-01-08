function G = bprop_net6(net, input, internal, output, W)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    antembed_bias = isfield(net, 'antembed_bias') && net.antembed_bias;
    srcembed_bias = isfield(net, 'srcembed_bias') && net.srcembed_bias;
    
    G = struct();

    % The last output of the softmax is unconnected
    output = output(:,1:end-1);
    targets = input.targets(:,1:end-1);

    outlayer_inputgrads = output - targets;
    
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
    
    Ares_inputgrads = internal.Ares .* (1 - internal.Ares) .* ...
        sum(internal.antfeatures .* wantfeatures_grads(input.antmap,:), 2);
    G.AhidAres = addbias(internal.Ahid)' * Ares_inputgrads;

    Ahid_inputgrads = internal.Ahid .* (1 - internal.Ahid) .* ...
        (Ares_inputgrads * W.AhidAres(2:end,:)');
    G.linkAhid = addbias(input.link)' * Ahid_inputgrads;

    if isfield(net, 'regulariser') && net.regulariser > 0
        G = transform_weights(@(x,y) x + net.regulariser * y, G, W);
    end

    if isfield(net, 'l1regulariser') && net.l1regulariser > 0
        G = transform_weights(@(x,y) x + net.l1regulariser * sign(y), G, W);
    end
end

