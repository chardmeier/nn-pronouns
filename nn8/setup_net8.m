function net = setup_net8(srcembed, srcjoin, betasensors, Ahid1, Ahid2, Lhid, hidden, output, vocab)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    logistictransfer = struct('f', @(x) 1 ./ (1 + exp(-x)), ...
        'df', @(x) x .* (1 - x));
    
    net = struct();
    
    net.transfer = struct('srcembed', logistictransfer, ...
        'srcjoin', logistictransfer, 'Ahid1', logistictransfer, ...
        'Ahid2', logistictransfer, 'Lhid', logistictransfer, ...
        'hidden', logistictransfer);
    
    net.link = sum(vocab.activelinkfeat);
    net.srcwvec = size(vocab.srcwvecs, 2) + 1; % add bias
    net.antwvec = size(vocab.antwvecs, 2) + 1; % add bias
    net.srcembed = srcembed;
    net.srcjoin = srcjoin;
    net.betasensors = betasensors;
    net.Ahid1 = Ahid1;
    net.Ahid2 = Ahid2;
    net.Lhid = Lhid;
    net.hidden = hidden;
    net.output = output;

    net.sample_antfeatures = 0;
    
    net.srcvoc = vocab.srcvoc;
    net.tgtvoc = vocab.tgtvoc;
    net.srcsingle = vocab.srcsingle;
    net.tgtsingle = vocab.tgtsingle;
    net.srcwvecs = addbias(vocab.srcwvecs);
    net.antwvecs = addbias(vocab.antwvecs);
    
    net.srcprons = length(vocab.srcprons);
    net.srcngsize = vocab.srcngsize;
    
    net.regulariser = 1e-3;
    
    net.nweights = net.srcwvec * net.srcembed + ...
        (net.srcprons + net.srcngsize * net.srcembed + 1) * net.srcjoin + ...
        (net.link + 1) * net.Lhid + (net.Lhid + 1) * 1 + ...
        net.betasensors * 3 + net.antwvec * net.Ahid1 + (net.Ahid1 + 1) * net.Ahid2 + ...
        (net.srcjoin + net.Ahid2 + 2) * net.hidden + (net.hidden + 2) * (net.output - 1);
end

