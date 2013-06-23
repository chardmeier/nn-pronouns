function net = setup_net3(srcembed, antembed, hidden, vocab, tgtprons)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    net = struct();
    net.srcembed = srcembed;
    net.antembed = antembed;
    net.hidden = hidden;
    net.output = length(tgtprons);
    net.targetpronouns = tgtprons;
    net.regulariser = 0;
    net.srcvoc = vocab.srcvoc;
    net.tgtvoc = vocab.tgtvoc;
    
    net.srcngsize = vocab.srcngsize;
    net.nweights = length(net.srcvoc) * net.srcembed + net.srcngsize * net.srcembed * net.hidden + ...
        length(net.tgtvoc) * net.antembed + net.antembed * net.hidden + net.hidden * net.output + ...
        2 * net.hidden + 2 * net.output; % The last ones are the biases and the NADA inputs
end

