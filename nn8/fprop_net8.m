function [output, internal] = fprop_net8(varargin)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    if nargin < 3 || nargin > 5
        error('Wrong number of parameters for fprop_net8.');
    end
    net = varargin{1};
    input = varargin{2};
    W = varargin{3};
    if nargin > 3
        prediction_mode = varargin{4};
    else
        prediction_mode = true;
    end
    if nargin > 4
        config = varargin{5};
    else
        config = struct('togpu', @double, 'fromgpu', @double, 'gpuzeros', @zeros, 'gpurand', @rand);
    end
    
    togpu = config.togpu;
    fromgpu = config.fromgpu;
    gpuzeros = config.gpuzeros;
    gpurand = config.gpurand;
    
    dropout = 0;
    if isfield(net, 'dropout')
        if prediction_mode
            W.srcembjoin = (1 - net.dropout) * W.srcembjoin;
            W.linkLhid = (1 - net.dropout) * W.linkLhid;
            W.LhidLres = (1 - net.dropout) * W.LhidLres;
            W.Ahid1Ahid2 = (1 - net.dropout) * W.Ahid1Ahid2;
            W.joinhid = (1 - net.dropout) * W.joinhid;
            W.hidout = (1 - net.dropout) * W.hidout;
            dropout = 0;
        else
            dropout = net.dropout;
        end
    end
    function y = do_dropout(x)
        if dropout > 0
            y = (rand(size(x)) < dropout) .* x;
        else
            y = x;
        end
    end
    
    if ~prediction_mode && isfield(net, 'dropout_src')
        drop = randperm(input.nitems, round(net.dropout_src * input.nitems));
    else
        drop = [];
    end
    
    srcembed = gpuzeros(input.nitems, net.srcembed, net.srcngsize);
    xwvec = cell(net.srcngsize, 1);
    for i = 1:net.srcngsize
        xwvec{i} = togpu(net.srcwvecs(input.src(:,i),:));
        xwvec{i}(drop,:) = 0;
        srcembed(:,:,i) = xwvec{i} * W.srcembed;
    end
    srcembed(:) = net.transfer.srcembed.f(srcembed(:));

    srcprons = bsxfun(@eq, input.srcprons, 1:net.srcprons);
    srcprons(drop,:) = 0;
    srcembcomplete = [srcprons, do_dropout(reshape(srcembed, input.nitems, net.srcngsize * net.srcembed))];
    srcjoin = net.transfer.srcjoin.f(addbias(srcembcomplete) * W.srcembjoin);
    
    antwvec = net.antwvecs(input.ant(:,2),:);
    Ahid1 = net.transfer.Ahid1.f(antwvec * W.antembed);
    [betamap,betaderiv] = betasensor(W.betasensors, input.ant(:,1));
    Ahid1agg = addbias(do_dropout(reshape((betamap * Ahid1)', net.betasensors * net.Ahid1, [])'));
    Ahid2 = net.transfer.Ahid2.f(Ahid1agg * W.Ahid1Ahid2);
    if net.sample_antfeatures
        santfeatures = 0 + (gpurand(size(Ahid2)) < Ahid2);
    else
        santfeatures = Ahid2;
    end
    
    Lhid = net.transfer.Lhid.f(addbias(do_dropout(togpu(input.link))) * W.linkLhid);
    Lresin = addbias(do_dropout(Lhid)) * W.LhidLres;
    Lres = groupwise_softmax(Lresin, input.antmap);
    %Lresc = cellfun(@(x) softmax(Lresin(x,:)')', input.antidx, 'UniformOutput', false);
    %Lres = vertcat(Lresc{:});
    %Lres = sigmoid(addbias(Lhid) * W.LhidLres);
    
    wLres = togpu(sparse(input.antmap, 1:length(input.antmap), fromgpu(Lres)));
    wantfeatures = wLres * santfeatures;
    
    nada = togpu(full(input.nada));
    join = do_dropout([srcjoin, wantfeatures]);
    
    hidden = do_dropout(net.transfer.hidden.f(addbias([nada, join]) * W.joinhid));
    output = softmax(addbias([nada, hidden]) * W.hidout, 'addcategory'); 

    internal = struct('Lhid', Lhid, 'Lres', Lres, 'wLres', wLres, 'srcembcomplete', srcembcomplete, ...
        'wvec', {xwvec}, 'srcembed', srcembed, 'antwvec', antwvec, 'Ahid1', Ahid1, 'Ahid1agg', Ahid1agg, ...
        'betamap', betamap, 'betaderiv', betaderiv, 'Ahid2', Ahid2, 'join', join, 'hidden', hidden, 'nada', nada);
end

