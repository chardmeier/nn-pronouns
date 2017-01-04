function net = setup_net6(srcembed, antembed, Ahid, hidden, output, vocab)
% Set up a network configuration (layer sizes and vocabularies).

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

