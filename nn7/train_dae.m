function [best_fweights, trainerr, valerr, best_bweights] = train_dae(id, input, cval, nhidden)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    nsteps = 100;
    stddev = .01;
    momentum = 0.9;
    alpha = .0005;
    alphadecay = 1;
    batchsize = 30;
    adjust_rate = true;
    early_stopping = true;
    noise_threshold = .5;
    l2regulariser = 1e-3;
    
    nitems = size(input, 1);
    nvis = size(input, 2);
    
    if exist('gpuDeviceCount', 'file') && gpuDeviceCount >= 1
        gpu = gpuDevice;
        if gpu.DeviceSupported
            togpu = @gpuArray;
            fromgpu = @gather;
            gpurand = @gpuArray.rand;
        else
            togpu = @double;
            fromgpu = @double;
            gpurand = @rand;
        end
    else
        togpu = @double;
        fromgpu = @double;
        gpurand = @rand;
    end
    
    val = togpu(cval);
    
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
    
    best_valerr = inf;
    alphachange_steps = 0;
    tiny = 1e-30;
    indices = [1:nitems, zeros(1, npad)];
    for i = 1:nsteps
        batchperm = reshape(indices(randperm(numel(indices))), nbatch, batchsize);
        
        err = 0;
        for j = 1:nbatch
            if mod(j,100) == 0
                fprintf('.');
            end
            thisperm = batchperm(j, batchperm(j,:) > 0);
            batchinput = togpu(input(thisperm,:));
            
            noisyinput = (gpurand(size(batchinput)) < noise_threshold) .* batchinput;
            
            % Forward propagation
            hidden = sigmoid(addbias(noisyinput) * fweights);
            output = sigmoid(addbias(hidden) * bweights);
            
            % Backpropagation
            error = output - batchinput;
            bgrads = addbias(hidden)' * error;
            fgrads = addbias(noisyinput)' * (hidden .* (1 - hidden) .* (error * bweights(2:end,:)'));

            err = err - sum(sum(batchinput .* log(max(tiny, output)))) / nitems;
            
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
        trainerr(i) = err;
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
        
        noisyinput = (gpurand(size(val)) < noise_threshold) .* val;
        hidden = sigmoid(addbias(noisyinput) * fweights);
        valout = sigmoid(addbias(hidden) * bweights);

        valerr(i) = -sum(sum(val .* log(max(tiny, valout)))) / size(val, 1);
        if(valerr(i) < best_valerr)
            best_valerr = valerr(i);
            best_fweights = fromgpu(fweights);
            best_bweights = fromgpu(bweights);
        end
        fprintf('\n%d (%g): Training error: %g, validation error: %g\n', i, alpha, trainerr(i), valerr(i));
        
        if early_stopping && i > 20 && min(valerr(i-19:i)) > best_valerr && valerr(i) - best_valerr > .1
            break
        end
        
        if mod(i, 10) == 0
            save(sprintf('dae-%s.%d.mat', id, i));
        end
    end
    
    valerr = valerr(1:i);
    trainerr = trainerr(1:i);
end

