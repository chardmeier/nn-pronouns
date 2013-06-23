function [batchinput, batchtargets] = create_batch_net3(input, targets, batchperm)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    batchinput = input;
    batchinput.nitems = length(batchperm);
    batchinput.ntrain = batchinput.nitems;
    batchinput.src = input.src(batchperm,:);
    batchinput.ant = input.ant(batchperm,:);
    batchinput.antweights = input.antweights(batchperm,:);
    batchinput.anaphwords = input.anaphwords(batchperm,:);
    batchinput.nada = input.nada(batchperm,:);
    
    batchtargets = targets(batchperm,:);
end

