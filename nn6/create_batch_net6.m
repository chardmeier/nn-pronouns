function batch = create_batch_net6(input, batchperm)
% Create batch from a subset of the input set.

    batchant = vertcat(input.antidx{batchperm});
    
    batch = struct();
    batch.nitems = length(batchperm);
    batch.src = input.src(batchperm,:);
    batch.ant = input.ant(batchant,:);
    batch.link = input.link(batchant,:);
    batch.targets = input.targets(batchperm,:);
    batch.nada = input.nada(batchperm,:);
    [batch.antmap,batch.antidx] = idxcell2map(input.antidx(batchperm));
end

