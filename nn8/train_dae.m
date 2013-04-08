function [best_fweights, trainerr, valerr, best_bweights] = train_dae(id, input, val, nhidden, params)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    nsteps = params.nsteps;
    stddev = params.stddev;
    momentum = params.momentum;
    alpha = params.alpha;
    alphadecay = params.alphadecay;
    batchsize = params.batchsize;
    adjust_rate = params.adjust_rate;
    early_stopping = params.early_stopping;
    noise_threshold = params.noise_threshold;
    l2regulariser = params.l2regulariser;
    
    F1 = params.F1;
    DF1 = params.DF1;
    F2 = params.F2;

    nitems = size(input, 1);
    nvis = size(input, 2);
    
    if params.use_gpu && exist('gpuDeviceCount', 'file') && gpuDeviceCount >= 1
        gpu = gpuDevice;
        if gpu.DeviceSupported
            togpu = @(x) gpuArray(single(x));
            fromgpu = @(x) double(gather(x));
            gpurand = @(varargin) gpuArray.rand(varargin{:}, 'single');
            gpuones = @(varargin) gpuArray.ones(varargin{:}, 'single');
        else
            togpu = @double;
            fromgpu = @double;
            gpurand = @rand;
            gpuones = @(varargin) ones(varargin{:});
        end
    else
        togpu = @double;
        fromgpu = @double;
        gpurand = @rand;
        gpuones = @(varargin) ones(varargin{:});
    end
    
    if mod(nitems, batchsize) > 0
        npad = batchsize - mod(nitems, batchsize);
    else
        npad = 0;
    end
    nbatch = (nitems + npad) / batchsize;
    fprintf('%d batches.\n', nbatch);
    
    trainerr = zeros(1, nsteps);
    valerr = zeros(1, nsteps);
    
    fweights = togpu(stddev * randn(nvis + 1, nhidden));
    bweights = togpu(stddev * randn(nhidden + 1, nvis));
    fweights(1,:) = 0;
    bweights(1,:) = 0;
    
    fgain = togpu(ones(nvis + 1, nhidden));
    fweight_change = togpu(zeros(nvis + 1, nhidden));
    fprev_grad = togpu(zeros(nvis + 1, nhidden));
    frms = togpu(ones(nvis + 1, nhidden));
    
    bgain = togpu(ones(nhidden + 1, nvis));
    bweight_change = togpu(zeros(nhidden + 1, nvis));
    bprev_grad = togpu(zeros(nhidden + 1, nvis));
    brms = togpu(ones(nhidden + 1, nvis));
    
    dotsteps = floor(nbatch / 80);
    
    best_valerr = inf;
    alphachange_steps = 0;
    tiny = 1e-30;
    indices = [1:nitems, zeros(1, npad)];
    for i = 1:nsteps
        batchperm = reshape(indices(randperm(numel(indices))), nbatch, batchsize);
        
        err = 0;
        for j = 1:nbatch
            if mod(j,dotsteps) == 0
                fprintf('.');
            end
            thisperm = batchperm(j, batchperm(j,:) > 0);
            currentsize = length(thisperm);
            batchinput = togpu(input(thisperm,:));
            
            noisyinput = gpuones(currentsize, nvis + 1);
            noisyinput(:,2:end) = (gpurand(size(batchinput)) < noise_threshold) .* batchinput;
            
            % Forward propagation
            hidden = gpuones(currentsize, nhidden + 1);
            hidden(:,2:end) = F1(noisyinput * fweights);
            output = F2(hidden * bweights);
            
            % Backpropagation
            error = output - batchinput;
            bgrads = hidden' * error;
            fgrads = noisyinput' * (DF1(hidden(:,2:end)) .* (error * bweights(2:end,:)'));

            %err = err - sum(sum(batchinput .* log(max(tiny, output)))) / nitems;
            err = err + sum(sum(error .^ 2)) / nitems;
            
            fgrads(2:end,:) = fgrads(2:end,:) + l2regulariser * fweights(2:end,:);
            bgrads(2:end,:) = bgrads(2:end,:) + l2regulariser * bweights(2:end,:);
            
            frms = 0.9 * frms + 0.1 * fgrads .^ 2;
            brms = 0.9 * brms + 0.1 * bgrads .^ 2;

            fgrads = fgrads ./ sqrt(frms + tiny);
            bgrads = bgrads ./ sqrt(brms + tiny);
            
            fweight_change = momentum * fweight_change - alpha * fgain .* fgrads;
            bweight_change = momentum * bweight_change - alpha * bgain .* bgrads;
            
            fweights = fweights + fweight_change;
            bweights = bweights + bweight_change;
            
            fchanged_signs = (fgrads < 0 & fprev_grad > 0) | (fgrads > 0 & fprev_grad < 0);
            fgain(fchanged_signs) = .95 * fgain(fchanged_signs);
            fsame_signs = (fgrads < 0 & fprev_grad < 0) | (fgrads > 0 & fprev_grad > 0);
            fgain(fsame_signs) = fgain(fsame_signs) + .05;
            
            bchanged_signs = (bgrads < 0 & bprev_grad > 0) | (bgrads > 0 & bprev_grad < 0);
            bgain(bchanged_signs) = .95 * bgain(bchanged_signs);
            bsame_signs = (bgrads < 0 & bprev_grad < 0) | (bgrads > 0 & bprev_grad > 0);
            bgain(bsame_signs) = bgain(bsame_signs) + .05;
            
            fprev_grad = fgrads;
            bprev_grad = bgrads;
        end
        trainerr(i) = fromgpu(err);
        if adjust_rate && i > 6 && sum(diff(trainerr((i-6):(i-1))) > 0) > 2 && alphachange_steps > 5
            %alpha = alpha / 2;
            alpha = alpha * .8;
            fprintf('Decreasing learning rate to %g\n', alpha);
            alphachange_steps = 0;
        end
        if adjust_rate && i > 6 && alphachange_steps > 5
            prob = .3 * sum(diff(trainerr((i-6):(i-1))) < 0) / 6;
            if rand < prob
                alpha = alpha * 1.05;
                %alpha = alpha + .0001;
                alphachange_steps = 0;
                fprintf('Increasing learning rate to %g\n', alpha);
            end
        end
        alpha = alphadecay * alpha;
        
        alphachange_steps = alphachange_steps + 1;
        
        cfweights = fromgpu(fweights);
        cbweights = fromgpu(bweights);

        noisyinput = (rand(size(val)) < noise_threshold) .* val;
        valhidden = F1(addbias(noisyinput) * cfweights);
        valout = F2(addbias(valhidden) * cbweights);

        %valerr(i) = fromgpu(-sum(sum(val .* log(max(tiny, valout)))) / size(val, 1));
        valerr(i) = sum(sum((valout - val) .^ 2)) / size(val, 1);
        if(valerr(i) < best_valerr)
            best_valerr = valerr(i);
            best_fweights = cfweights;
            best_bweights = cbweights;
        end
        fprintf('\n%d (%g): Training error: %g, validation error: %g\n', i, alpha, trainerr(i), valerr(i));
        
        if early_stopping && i > 20 && min(valerr(i-19:i)) > best_valerr && valerr(i) - best_valerr > .1
            break
        end
        
        if mod(i, 10) == 0
            save(sprintf('dae-%s.%d.mat', id, i), '-v7.3', 'best_fweights', 'best_bweights');
        end
    end
    
    valerr = valerr(1:i);
    trainerr = trainerr(1:i);
end

