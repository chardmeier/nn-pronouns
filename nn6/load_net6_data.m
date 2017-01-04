function [input, vocab] = load_net6_data(dataprefix, trainidx, validx, testidx)
% Load data set from disk.

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
    
    vocab.srcsingle = load(inpfile('srcsingle'));
    vocab.tgtsingle = load(inpfile('tgtsingle'));
        
    vocab.srcngsize = 7;
    
    allsrc = spconvert([load(inpfile('srcfeat')); 1 ((vocab.srcngsize+1) * length(vocab.srcvoc) - 1) 0]);
    allant = spconvert([load(inpfile('antfeat')); 1 length(vocab.tgtvoc) 0]);
    alltargets = load(inpfile('targets'));
    allantmap = load(inpfile('antmap'));
    
    alllink = spconvert(load(inpfile('linkfeat')));
    
    nexamples = size(allsrc, 1);
    
    range = (1:length(allantmap))';
    map = sparse(range, allantmap, true);
    allantidx = cellfun(@(x) range(logical(map(:,x))), num2cell(1:nexamples), ...
        'UniformOutput', false);

    allnada = load(inpfile('nada'));

    trainant = vertcat(allantidx{trainidx});
    trainlink = alllink(trainant,:);
    lfratio = sum(trainlink, 1) / size(trainlink, 1);
    vocab.activelinkfeat = lfratio > .01 & lfratio < .99;
    
    input.nitems = length(trainidx);
    input.src = allsrc(trainidx,:);
    input.ant = allant(trainant,:);
    input.link = alllink(trainant,vocab.activelinkfeat);
    input.nada = allnada(trainidx);
    input.targets = alltargets(trainidx,:);
    [input.antmap,input.antidx] = idxcell2map(allantidx(trainidx));
    
    if length(validx) > 0
        valant = vertcat(allantidx{validx});
        input.val = struct();
        input.val.nitems = length(validx);
        input.val.src = allsrc(validx,:);
        input.val.ant = allant(valant,:);
        input.val.link = alllink(valant,vocab.activelinkfeat);
        input.val.targets = alltargets(validx,:);
        input.val.nada = allnada(validx);
        [input.val.antmap,input.val.antidx] = idxcell2map(allantidx(validx));
    end
    
    if length(testidx) > 0
        testant = vertcat(allantidx{testidx});
        input.test = struct();
        input.test.nitems = length(testidx);
        input.test.src = allsrc(testidx,:);
        input.test.ant = allant(testant,:);
        input.test.link = alllink(testant,vocab.activelinkfeat);
        input.test.nada = allnada(testidx);
        input.test.targets = alltargets(testidx,:);
        [input.test.antmap,input.test.antidx] = idxcell2map(allantidx(testidx));
    end
end

