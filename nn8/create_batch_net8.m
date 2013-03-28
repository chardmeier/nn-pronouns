function batch = create_batch_net8(input, batchperm)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    batchant = vertcat(input.antidx{batchperm});
    
    batch = struct();
    batch.nitems = length(batchperm);
    batch.src = input.src(batchperm,:);
    batch.srcprons = input.srcprons(batchperm,:);
    
    antwordmap = sparse(batchant, 1, 1:length(batchant), size(input.antmap, 1), 1);
    batch.ant = input.ant(antwordmap(input.ant(:,1)) ~= 0,:);
    unsortedant = antwordmap(batch.ant(:,1));
    [~,idx] = sort(unsortedant(:,1));
    batch.ant(:,1) = unsortedant(idx,:);
    
    batch.link = input.link(batchant,:);
    batch.targets = input.targets(batchperm,:);
    batch.nada = input.nada(batchperm,:);
    [batch.antmap,batch.antidx] = idxcell2map(input.antidx(batchperm));
end