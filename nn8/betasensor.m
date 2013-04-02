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
    
    alpha = repmat(exp(params(:,2))', length(pointvec), 1);
    beta = repmat(exp(params(:,3))', length(pointvec), 1);
    
    ibd = inbeder(pointvec(:), alpha(:), beta(:));
    
    cumtable = ones(maxantwords, maxantwords, nsensors);
    cumtable(repmat(tril(true(maxantwords), -1), [1 1 nsensors])) = ibd(:,1);
    ptable = diff([zeros(maxantwords, 1, nsensors), cumtable], 1, 2);
    ftable = squeeze(sum(bsxfun(@times, permute(params(:,1), [3 2 1]), ptable), 3));
    
    indices = sub2ind(size(ftable), blocks(mapvec), counters);
    factors = ftable(indices);
    
    if isa(mapvec, 'gpuArray')
        out = gpuArray.zeros(nout, nin);
        out(sub2ind(size(out), mapvec, 1:nin)) = factors;
        %mapmat = bsxfun(@eq, 1:nout, mapvec);
        %tout = mapmat;
        %tout(mapmat) = factors;
        %out = tout';
    else
        out = sparse(mapvec, 1:nin, factors, nout, nin);
    end
    
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
        deriv.wdavals = bsxfun(@times, (params(:,1) .* exp(params(:,2)))', davals);
        
        dbcumtable = zeros(maxantwords, maxantwords, nsensors);
        dbcumtable(repmat(tril(true(maxantwords), -1), [1 1 nsensors])) = ibd(:,4);
        dbtable = diff([zeros(maxantwords, 1, nsensors), dbcumtable], 1, 2);
        dbvals = reshape(dbtable(sub2ind(size(dbtable), idx1, idx2, idx3(:))), nin, nsensors);
        deriv.wdbvals = bsxfun(@times, (params(:,1) .* exp(params(:,3)))', dbvals);
    end
end

