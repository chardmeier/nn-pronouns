function batch = create_batch_deplm1(input, batchperm)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    batch = struct();
    batch.nonvoc = input.nonvoc(batchperm,:);
    batch.srchead = input.srchead(batchperm,:);
    batch.srcdep = input.srcdep(batchperm,:);
    batch.tgthead = input.tgthead(batchperm,:);
    batch.tgtdep = input.tgtdep(batchperm,:);
    batch.nitems = size(batch.nonvoc, 1);
end

