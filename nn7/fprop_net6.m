function [output, internal] = fprop_net6(net, input, W, prediction_mode)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    srcembed_bias = isfield(net, 'srcembed_bias') && net.srcembed_bias;
    
    srcvocsize = length(net.srcvoc);
    srcembed = zeros(input.nitems, net.srcembed, net.srcngsize);
    for i = 1:net.srcngsize
        wvec = input.src(:,((i-1)*srcvocsize+1):(i*srcvocsize));
        if srcembed_bias
            wvec = addbias(wvec);
        end
        if ~prediction_mode && isfield(net, 'dropout_src')
            wvec = spfun(@(x) x * (rand < net.dropout_src), wvec);
        end
        srcembed(:,:,i) = wvec * W.srcembed;
    end
    srcembed(:) = sigmoid(srcembed(:));

    if isfield(net, 'antembed_bias') && net.antembed_bias
        ant = addbias(input.ant);
    else
        ant = input.ant;
    end
    antfeatures = sigmoid(ant * W.antembed);
    if isfield(net, 'sample_antfeatures') && net.sample_antfeatures
        santfeatures = 0 + (rand(size(antfeatures)) < antfeatures);
    else
        santfeatures = antfeatures;
    end
    
    Ahid = sigmoid(addbias(input.link) * W.linkAhid);

    Aresin = addbias(Ahid) * W.AhidAres;
    Aresc = cellfun(@(x) softmax(Aresin(x,:)')', input.antidx, 'UniformOutput', false);
    Ares = vertcat(Aresc{:});
    %Ares = sigmoid(addbias(Ahid) * W.AhidAres);
    
    wAres = sparse(input.antmap, 1:length(input.antmap), Ares);
    wantfeatures = wAres * santfeatures;

    embed = [reshape(srcembed, input.nitems, net.srcngsize * net.srcembed), ...
             wantfeatures];
    if isfield(net, 'sample_embed') && net.sample_embed
        sembed = 0 + (rand(size(embed)) < embed);
    else
        sembed = embed;
    end
    
    hidden = sigmoid(addbias([input.nada, sembed]) * W.embhid);
    output = softmax(addbias([input.nada, hidden]) * W.hidout, 'addcategory'); 

    internal = struct('Ahid', Ahid, 'Ares', Ares, 'wAres', wAres, 'antfeatures', antfeatures, ...
        'embed', embed, 'hidden', hidden);
end

