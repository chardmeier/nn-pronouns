function net = setup_net7(srcembed, antembed, Ahid, hidden, output, vocab)
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
    net.srcembed = srcembed;
    net.antembed = antembed;
    net.Ahid = Ahid;
    net.hidden = hidden;
    net.output = output;

    net.srcvoc = vocab.srcvoc;
    net.tgtvoc = vocab.tgtvoc;
    net.srcsingle = vocab.srcsingle;
    net.tgtsingle = vocab.tgtsingle;
    
    net.srcngsize = vocab.srcngsize;
end

