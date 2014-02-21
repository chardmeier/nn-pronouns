function net = setup_deplm1(vocsize, vocembed, hidden, output, vocab)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    net = struct();
    net.vocsize = vocsize;
    net.vocembed = vocembed;
    net.hidden = hidden;
    net.output = output;

    net.srcvoc = vocab.srcvoc;
    net.tgtvoc = vocab.tgtvoc;
end

