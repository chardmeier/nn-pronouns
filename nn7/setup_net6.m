function net = setup_net6(srcembed, antembed, Ahid, hidden, output, vocab)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    net = struct();
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

