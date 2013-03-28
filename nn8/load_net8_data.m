function [input, vocab] = load_net8_data(dataprefix, trainidx, validx, testidx)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    function name = inpfile(ext)
        name = strcat(dataprefix, '.', ext);
    end

    input = struct();
    vocab = struct();
    
    id = fopen(inpfile('srcvoc'));
    r = textscan(id, '%s');
    vocab.srcvoc = r{1};
    fclose(id);
    
    id = fopen(inpfile('tgtvoc'));
    r = textscan(id, '%s');
    vocab.tgtvoc = r{1};
    fclose(id);
    
    %vocab.srcsingle = load(inpfile('srcsingle'));
    %vocab.tgtsingle = load(inpfile('tgtsingle'));
    vocab.srcsingle = [];
    vocab.tgtsingle = [];

    ssrcwvecs = load(inpfile('srcwvecs'));
    vocab.srcwvecs = full(spconvert(ssrcwvecs));
    
    santwvecs = load(inpfile('tgtwvecs'));
    vocab.antwvecs = full(spconvert(santwvecs));

    vocab.srcngsize = 6;
    rawsrc = load(inpfile('src'));
    % Special treatment for the actual pronoun
    allsrc = rawsrc(:,[1:3 5:7]);
    vocab.srcprons = unique(rawsrc(:,4));
    [~,allsrcprons] = ismember(rawsrc(:,4), vocab.srcprons);
    
    allant = load(inpfile('ant'));
    alltargets = load(inpfile('targets'));
    allantmap = load(inpfile('antmap'));
    
    % HACK HACK HACK!
    noant = setdiff(1:length(allantmap), allant(:,1));
    xant = [allant; [noant(:), repmat(length(vocab.tgtvoc), length(noant), 1)]];
    [~,idx] = sort(xant(:,1));
    allant = xant(idx,:);
    
    alllink = spconvert(load(inpfile('linkfeat')));
    
    nexamples = size(allsrc, 1);
    
    range = (1:length(allantmap))';
    map = sparse(range, allantmap, true);
    allantidx = cellfun(@(x) range(logical(map(:,x))), num2cell(1:nexamples), ...
        'UniformOutput', false);

    allnada = sparse(nexamples, 1);

    trainant = vertcat(allantidx{trainidx});
    trainlink = alllink(trainant,:);
    lfratio = sum(trainlink, 1) / size(trainlink, 1);
    vocab.activelinkfeat = lfratio > .01 & lfratio < .99;
    
    allinput.nitems = nexamples;
    allinput.src = allsrc;
    allinput.srcprons = allsrcprons;
    allinput.ant = allant;
    allinput.link = alllink(:,vocab.activelinkfeat);
    allinput.nada = allnada;
    allinput.targets = alltargets;
    allinput.antidx = allantidx;
    allinput.antmap = idxcell2map(allantidx);
    
    input = create_batch_net8(allinput, trainidx);
    input.val = create_batch_net8(allinput, validx);
    input.test = create_batch_net8(allinput, testidx);
    
%     input.nitems = length(trainidx);
%     input.src = allsrc(trainidx,:);
%     input.srcprons = allsrcprons(trainidx,:);
%     input.ant = allant(ismember(allant(:,1), trainant),:);
%     input.link = alllink(trainant,vocab.activelinkfeat);
%     input.nada = allnada(trainidx);
%     input.targets = alltargets(trainidx,:);
%     [input.antmap,input.antidx] = idxcell2map(allantidx(trainidx));
%     
%     valant = vertcat(allantidx{validx});
%     input.val = struct();
%     input.val.nitems = length(validx);
%     input.val.src = allsrc(validx,:);
%     input.val.srcprons = allsrcprons(validx,:);
%     input.val.ant = allant(ismember(allant(:,1), valant),:);
%     input.val.link = alllink(valant,vocab.activelinkfeat);
%     input.val.targets = alltargets(validx,:);
%     input.val.nada = allnada(validx);
%     [input.val.antmap,input.val.antidx] = idxcell2map(allantidx(validx));
%     
%     testant = vertcat(allantidx{testidx});
%     input.test = struct();
%     input.test.nitems = length(testidx);
%     input.test.src = allsrc(testidx,:);
%     input.test.srcprons = allsrcprons(testidx,:);
%     input.test.ant = allant(ismember(allant(:,1), testant),:);
%     input.test.link = alllink(testant,vocab.activelinkfeat);
%     input.test.nada = allnada(testidx);
%     input.test.targets = alltargets(testidx,:);
%     [input.test.antmap,input.test.antidx] = idxcell2map(allantidx(testidx));
end

