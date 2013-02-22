function batch = create_batch_net7(input, batchperm)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

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

