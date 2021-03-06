function [out,deriv] = betasensor(params, mapvec)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    nsensors = size(params, 1);
    nin = length(mapvec);
    nout = max(mapvec);
    
    sum_accumulator = [mapvec, ones(nin, 1)];
    blocks = accumarray(sum_accumulator, ones(nin, 1));
    cumblocks = cumsum([0; blocks]);
    counters = (1:nin)' - cumblocks(mapvec);
    
    maxantwords = 10;
    points = bsxfun(@times, 1:maxantwords, 1 ./ (1:maxantwords)');
    pointvec = repmat(points(tril(true(size(points)), -1)), 1, nsensors);
    
    alpha = repmat(exp(params(:,1))', length(pointvec), 1);
    beta = repmat(exp(params(:,2))', length(pointvec), 1);
    
    ibd = inbeder(pointvec(:), alpha(:), beta(:));
    
    cumtable = ones(maxantwords, maxantwords, nsensors);
    cumtable(repmat(tril(true(maxantwords), -1), [1 1 nsensors])) = ibd(:,1);
    ptable = diff([zeros(maxantwords, 1, nsensors), cumtable], 1, 2);
    
    if isa(mapvec, 'gpuArray')
        celltype = @gpuArray;
        bdcall = @blkdiag;
    else
        celltype = @sparse;
        bdcall = @blkdiagmex;
    end
    
    pcell = cell(maxantwords, 1);
    for i = 1:maxantwords
        pcell{i} = celltype(reshape(ptable(i,1:i,:), i, nsensors)');
    end
    out = bdcall(pcell{blocks});
    
    if nargout > 1
        deriv = struct();
        idx1 = repmat(blocks(mapvec), nsensors, 1);
        idx2 = repmat(counters, nsensors, 1);
        idx3 = repmat(1:nsensors, nin, 1);
        deriv.pvals = reshape(ptable(sub2ind(size(ptable), idx1, idx2, idx3(:))), nin, nsensors);
        
        dacumtable = zeros(maxantwords, maxantwords, nsensors);
        dacumtable(repmat(tril(true(maxantwords), -1), [1 1 nsensors])) = ibd(:,2);
        datable = diff([zeros(maxantwords, 1, nsensors), dacumtable], 1, 2);
        davals = reshape(datable(sub2ind(size(datable), idx1, idx2, idx3(:))), nin, nsensors);
        deriv.wdavals = bsxfun(@times, exp(params(:,1))', davals);
        
        dbcumtable = zeros(maxantwords, maxantwords, nsensors);
        dbcumtable(repmat(tril(true(maxantwords), -1), [1 1 nsensors])) = ibd(:,4);
        dbtable = diff([zeros(maxantwords, 1, nsensors), dbcumtable], 1, 2);
        dbvals = reshape(dbtable(sub2ind(size(dbtable), idx1, idx2, idx3(:))), nin, nsensors);
        deriv.wdbvals = bsxfun(@times, exp(params(:,2))', dbvals);
    end
end

