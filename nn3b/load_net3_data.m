function [input, vocab] = load_net3_data(file, testfile)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    input = struct();
    vocab = struct();
    
    id = fopen(file);
    raw = textscan(id, '%s %s %s %s %f', 'delimiter', ' ');
    fclose(id);
    
    allanaphwords = regexp(raw{1}, '\|', 'split');
    srcwords = regexp(raw{2}, '\|', 'split');
    antwords = regexp(raw{3}, '\|', 'split');
    allantweights = cellfun(@(x) cell2mat(cellfun(@str2double, x, 'UniformOutput', false)), ...
        regexp(raw{4}, '\|', 'split'), 'UniformOutput', false);
    allnada = raw{5};
    
    emptyants = strcmp(raw{3}, '');
    [antwords{emptyants}] = deal({'***NONE***'});
    [allantweights{emptyants}] = deal(1);
    
    vocab.srcvoc = unique([srcwords{:}, '***UNKNOWN***']);
    vocab.tgtvoc = unique([antwords{:}, '***UNKNOWN***']);
    srcvocmap = containers.Map(vocab.srcvoc, 1:length(vocab.srcvoc));
    tgtvocmap = containers.Map(vocab.tgtvoc, 1:length(vocab.tgtvoc));
    
    allsrc = cell2mat(cellfun(@(x) cell2mat(values(srcvocmap, x)), srcwords, 'UniformOutput', false));
    allant = cellfun(@(x) cell2mat(values(tgtvocmap, x)), antwords, 'UniformOutput', false);
    
    nexamples = size(allsrc, 1);
    input.ntrain = round(.9 * nexamples);
    input.nitems = input.ntrain;
    input.nval = nexamples - input.ntrain;
    
    perm = randperm(nexamples);
    trainidx = perm(1:input.ntrain);
    validx = perm((input.ntrain + 1):end);
    
    input.src = allsrc(trainidx,:);
    input.ant = allant(trainidx,:);
    input.antweights = allantweights(trainidx,:);
    input.anaphwords = allanaphwords(trainidx,:);
    input.nada = allnada(trainidx,:);
    
    input.val = struct();
    input.val.nitems = input.nval;
    input.val.src = allsrc(validx,:);
    input.val.ant = allant(validx,:);
    input.val.antweights = allantweights(validx,:);
    input.val.anaphwords = allanaphwords(validx,:);
    input.val.nada = allnada(validx,:);
    
    % Load test data
    
    id = fopen(testfile);
    raw = textscan(id, '%s %s %s %s %f', 'delimiter', ' ');
    fclose(id);
    
    input.test = struct();
    
    input.test.anaphwords = regexp(raw{1}, '\|', 'split');
    srcwords = regexp(raw{2}, '\|', 'split');
    antwords = regexp(raw{3}, '\|', 'split');
    input.test.antweights = cellfun(@(x) cell2mat(cellfun(@str2num, x, 'UniformOutput', false)), ...
        regexp(raw{4}, '\|', 'split'), 'UniformOutput', false);
    input.test.nada = raw{5};
    
    emptyants = strcmp(raw{3}, '');
    [antwords{emptyants}] = deal({'***NONE***'});
    [input.test.antweights{emptyants}] = deal(1);
    
    function ids = lookupMap(map, vals)
        unk = ~isKey(map, vals);
        if sum(unk) > 0
            [vals{unk}] = deal('***UNKNOWN***');
        end
        ids = values(map, vals);
    end

    input.test.src = cell2mat(cellfun(@(x) cell2mat(lookupMap(srcvocmap, x)), srcwords, 'UniformOutput', false));
    input.test.ant = cellfun(@(x) cell2mat(lookupMap(tgtvocmap, x)), antwords, 'UniformOutput', false);
    
    input.test.nitems = size(input.test.src, 1);
    input.ntest = input.test.nitems;

    vocab.srcngsize = size(input.src, 2);
end

