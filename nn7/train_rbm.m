function cweights = train_rbm(id, data, val, nhid, params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    if params.use_gpu && exist('gpuDeviceCount', 'file') && gpuDeviceCount >= 1
        gpu = gpuDevice;
        if gpu.DeviceSupported
            togpu = @(x) gpuArray(single(x));
            fromgpu = @(x) double(gather(x));
            gpurand = @(varargin) gpuArray.rand(varargin{:}, 'single');
            gpuones = @(varargin) gpuArray.ones(varargin{:}, 'single');
            gpuzeros = @(varargin) gpuArray.zeros(varargin{:}, 'single');
        else
            togpu = @double;
            fromgpu = @double;
            gpurand = @rand;
            gpuones = @(varargin) ones(varargin{:});
            gpuzeros = @(varargin) zeros(varargin{:});
        end
    else
        togpu = @double;
        fromgpu = @double;
        gpurand = @rand;
        gpuones = @(varargin) ones(varargin{:});
        gpuzeros = @(varargin) zeros(varargin{:});
    end
    
    function S = sample_bernoulli(P)
        S = rand(size(P)) < P;
    end

    alpha = params.alpha;
    stddev = params.stddev;
    batchsize = params.batchsize;
    nsteps = params.nsteps;
    momentum = params.momentum;
    l2regulariser = params.l2regulariser;
    
    sparsity_target = params.sparsity_target;
    sparsity_decay = params.sparsity_decay;
    sparsity_cost = params.sparsity_cost;
    
    ndata = size(data, 1);
    nvis = size(data, 2);
    
    batches_per_epoch = ceil(ndata / batchsize);
    
    tiny = 1e-30;
    
    weights = togpu(stddev * randn(nvis + 1, nhid + 1));
    datap = mean(data, 1);
    weights(:,1) = [0; log(datap ./ (1 - datap) + tiny)'];
    weights(1,2:end) = log(sparsity_target / (1 - sparsity_target));
    
    sparsity_q = sparsity_target;
    
    bias = gpuones(batchsize, 1);
    weight_change = gpuzeros(size(weights));
    rms = gpuones(size(weights));
    %gain = ones(size(weights));
    
    nvalsub = min(size(val, 1), 100);
    testsubset = togpu(data(1:nvalsub,:));
    valsub = togpu(val(1:nvalsub,:));
    
    nupdates = 0;
    for i = 1:nsteps
        fprintf('Epoch %d\n', i);
        
        for j = 1:batches_per_epoch
            batchperm = randperm(ndata, batchsize);
            batch = togpu(data(batchperm,:));

            %batch = (rand(size(batch)) < 0.5) .* batch;

            phid1 = sigmoid([bias, batch] * weights(:,2:end));
            hid1 = sample_bernoulli(phid1);
            vis2 = sigmoid([bias, hid1] * weights(2:end,:)');
            hid2 = sigmoid([bias, vis2] * weights(:,2:end));

            grads = ([bias,batch]' * [bias,phid1] - [bias,vis2]' * [bias,hid2]) / batchsize;
            grads(2:end,2:end) = grads(2:end,2:end) - l2regulariser * weights(2:end,2:end);
            
            current_sparsity = mean([phid1(:); hid2(:)]);
            sparsity_q = sparsity_decay * sparsity_q + (1 - sparsity_decay) * current_sparsity;
            grads(:,2:end) = grads(:,2:end) - sparsity_cost * (sparsity_q - sparsity_target);
            
            rms = 0.9 * rms + 0.1 * grads .^ 2;
            grads = grads ./ sqrt(rms + tiny);
            
            %a = sum(abs(weight_change(:)));
            %b = sum(abs(weights(:)));
            %fprintf('Update: %g/%g = %g\n', a, b, a/b);
            
            weight_change = momentum * weight_change + alpha * grads;
            weights = weights + weight_change;
            
            nupdates = nupdates + 1;
            
            %fprintf('.');
            %if mod(nupdates, 50) == 0
                %fprintf('%d\n', nupdates);
            %end
            
            if mod(nupdates, 600) == 0
                trainerr = sum(sum((batch - vis2) .^ 2)) / batchsize;

                hid1 = sample_bernoulli(sigmoid(addbias(valsub) * weights(:,2:end)));
                vis2 = sigmoid(addbias(hid1) * weights(2:end,:)');
                valerr = sum(sum((valsub - vis2) .^ 2)) / nvalsub;
                
		cw = fromgpu(weights(2:end,2:end));
		cwc = fromgpu(weight_change(2:end,2:end));
                figure(1), clf, hist(cw(:),30);
                figure(2), clf, hist(cwc(:),30);
                Ftest = -sum(testsubset * weights(2:end,1)) - ...
                    sum(sum(log1p(exp(addbias(testsubset) * weights(:,2:end)))));
                Fval = -sum(valsub * weights(2:end,1)) - ...
                    sum(sum(log1p(exp(addbias(valsub) * weights(:,2:end)))));
                
                fprintf('%d.%d: trainerr = %g, valerr = %g, |grad| = %f, |W| = %f, gap = %g\n', ...
                    i, j, trainerr, valerr, norm(grads, 'fro'), norm(weights, 'fro'), Ftest - Fval);
            end
        end
        cweights = fromgpu(weights);
        save(sprintf('rbm-%s.%d.mat', id, i), 'cweights');
    end
end

