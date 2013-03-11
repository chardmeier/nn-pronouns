function [output, internal] = fprop_net7(net, input, W, prediction_mode)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    if ~prediction_mode && isfield(net, 'dropout_src')
        drop = randperm(input.nitems, round(net.dropout_src * input.nitems));
    end
    
    srcembed = zeros(input.nitems, net.srcembed, net.srcngsize);
    xwvec = cell(net.srcngsize);
    for i = 1:net.srcngsize
        wvec = net.srcwvecs(input.src(:,i),:);
        if net.srcwvec_bias
            xwvec{i} = addbias(wvec);
        else
            xwvec{i} = wvec;
        end
        if ~prediction_mode && isfield(net, 'dropout_src')
            xwvec{i}(drop,:) = 0;
        end
        srcembed(:,:,i) = xwvec{i} * W.srcembed;
    end
    srcembed(:) = net.transfer.srcembed.f(srcembed(:));

    srcjoin = net.transfer.srcjoin.f(addbias(reshape(srcembed, input.nitems, net.srcngsize * net.srcembed)) * W.srcembjoin);
    
    Ahid1 = net.transfer.Ahid1.f(addbias(input.ant) * W.antembed);
    Ahid2 = net.transfer.Ahid2.f(addbias(Ahid1) * W.Ahid1Ahid2);
    if net.sample_antfeatures
        santfeatures = 0 + (rand(size(Ahid2)) < Ahid2);
    else
        santfeatures = Ahid2;
    end
    
    Lhid = net.transfer.Lhid.f(addbias(input.link) * W.linkLhid);
    Lresin = addbias(Lhid) * W.LhidLres;
    Lresc = cellfun(@(x) softmax(Lresin(x,:)')', input.antidx, 'UniformOutput', false);
    Lres = vertcat(Lresc{:});
    %Lres = sigmoid(addbias(Lhid) * W.LhidLres);
    
    wLres = sparse(input.antmap, 1:length(input.antmap), Lres);
    wantfeatures = wLres * santfeatures;

    join = [srcjoin, wantfeatures];
    
    hidden = net.transfer.hidden.f(addbias([input.nada, join]) * W.joinhid);
    output = softmax(addbias([input.nada, hidden]) * W.hidout, 'addcategory'); 

    internal = struct('Lhid', Lhid, 'Lres', Lres, 'wLres', wLres, ...
        'wvec', {xwvec}, 'srcembed', srcembed, 'Ahid1', Ahid1, 'Ahid2', Ahid2, 'join', join, ...
        'hidden', hidden);
end

