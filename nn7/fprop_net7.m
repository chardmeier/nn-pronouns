function [output, internal] = fprop_net7(net, input, W, prediction_mode)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    if nargin < 3 || nargin > 5
        error('Wrong number of parameters for fprop_net7.');
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
        config = struct('togpu', @double, 'gpuzeros', @zeros);
    end
    
    togpu = config.togpu;
    gpuzeros = config.gpuzeros;
    
    if ~prediction_mode && isfield(net, 'dropout_src')
        drop = randperm(input.nitems, round(net.dropout_src * input.nitems));
    else
        drop = [];
    end
    
    srcembed = gpuzeros(input.nitems, net.srcembed, net.srcngsize);
    xwvec = cell(net.srcngsize);
    for i = 1:net.srcngsize
        wvec = net.srcwvecs(input.src(:,i),:);
        if net.srcwvec_bias
            xwvec{i} = togpu(addbias(wvec));
        else
            xwvec{i} = togpu(wvec);
        end
        xwvec{i}(drop,:) = 0;
        srcembed(:,:,i) = xwvec{i} * W.srcembed;
    end
    srcembed(:) = net.transfer.srcembed.f(srcembed(:));

    srcprons = bsxfun(@eq, input.srcprons, 1:net.srcprons);
    srcprons(drop,:) = 0;
    srcembcomplete = [srcprons, reshape(srcembed, input.nitems, net.srcngsize * net.srcembed)];
    srcjoin = net.transfer.srcjoin.f(addbias(srcembcomplete) * W.srcembjoin);
    
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

    internal = struct('Lhid', Lhid, 'Lres', Lres, 'wLres', wLres, 'srcembcomplete', srcembcomplete, ...
        'wvec', {xwvec}, 'srcembed', srcembed, 'Ahid1', Ahid1, 'Ahid2', Ahid2, 'join', join, ...
        'hidden', hidden);
end

