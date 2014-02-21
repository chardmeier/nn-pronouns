function [input, vocab] = load_deplm1_data(dataprefix, trainidx, validx, testidx)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    function name = inpfile(ext)
        name = strcat(dataprefix, '.', ext);
    end

    input = struct();
    vocab = struct();
    
%     id = fopen(inpfile('srcvoc'));
%     r = textscan(id, '%s');
%     vocab.srcvoc = r{1};
%     fclose(id);
%     
%     id = fopen(inpfile('tgtvoc'));
%     r = textscan(id, '%s');
%     vocab.tgtvoc = r{1};
%     fclose(id);
    
    all = struct();
    all.nonvoc = full(spconvert(load(inpfile('nonvoc'))));
    all.srchead = full(addbias(spconvert(load(inpfile('srchead')))));
    all.srcdep = full(addbias(spconvert(load(inpfile('srcdep')))));
    all.tgthead = full(addbias(spconvert(load(inpfile('tgthead')))));
    all.tgtdep = full(addbias(spconvert(load(inpfile('tgtdep')))));
    all.targets = load(inpfile('targets'));
    all.nitems = size(all.nonvoc, 1);

    input = create_batch_deplm1(all, trainidx);
    input.test = create_batch_deplm1(all, testidx);
    input.val = create_batch_deplm1(all, validx);
end

