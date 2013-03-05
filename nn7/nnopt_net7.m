function [best_weights, trainerr, valerr, best_valerr] = nnopt_net7(id, net, input, params)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    if exist('gpuDeviceCount', 'file') && gpuDeviceCount >= 1
        gpu = gpuDevice;
        if gpu.DeviceSupported
            togpu = @gpuArray;
            fromgpu = @gather;
        else
            togpu = @double;
            fromgpu = @double;
        end
    else
        togpu = @double;
        fromgpu = @double;
    end
    
    momentum = params.momentum;
    alpha = params.initialrate;
    alphadecay = params.ratedecay;
    batchsize = params.batchsize;

    if mod(input.nitems, batchsize) > 0
        npad = batchsize - mod(input.nitems, batchsize);
    else
        npad = 0;
    end
    nbatch = (input.nitems + npad) / batchsize;
    
    trainerr = zeros(1, params.nsteps);
    valerr = zeros(1, params.nsteps);
    
    if isfield(net, 'initweights')
        weights = net.initweights;
    else
        weights = params.stddev * randn(net.nweights, 1);
    end
    gweights = togpu(weights);
    
    gain = togpu(ones(net.nweights, 1));
    %weight_change = sparse(net.nweights, 1);
    %prev_grad = sparse(net.nweights, 1);
    weight_change = togpu(zeros(net.nweights, 1));
    prev_grad = togpu(zeros(net.nweights, 1));
    rms = togpu(ones(net.nweights, 1));
    best_valerr = inf;
    alphachange_steps = 0;
    tiny = 1e-30;
    indices = [1:input.nitems, zeros(1, npad)];
    for i = 1:params.nsteps
        batchperm = reshape(indices(randperm(numel(indices))), nbatch, batchsize);
        
        err = 0;
        for j = 1:nbatch
            thisperm = batchperm(j, batchperm(j,:) > 0);
            batchinput = create_batch_net7(input, thisperm);
            W = weightstruct_net7(net, weights);
            [output,hidden] = fprop_net7(net, batchinput, W, false);
            G = bprop_net7(net, batchinput, hidden, output, W);
            grad = togpu(weightvector_net7(G));
            err = err - sum(sum(batchinput.targets .* log(max(tiny, output)))) / input.nitems;
            
            grad = (grad > 1e-5) .* grad; % conserve sparsity
            if isfield(net, 'regulariser') && net.regulariser > 0
                grad = grad + net.regulariser * gweights;
            end
            if isfield(net, 'l1regulariser') && net.l1regulariser > 0
                grad = grad + net.l1regulariser * gweights;
            end
            
            rms = 0.9 * rms + 0.1 * grad .^ 2;
            %[a,~,x] = find(grad);
            %grad = sparse(a, 1, x ./ sqrt(rms(a) + tiny), net.nweights, 1);
            grad = grad ./ sqrt(rms + tiny);
            
            weight_change = momentum * weight_change - alpha * gain .* grad;
            gweights = weights + weight_change;
            weights = fromgpu(gweights);
            
            %if isfield(net, 'weightconstraint')
            %    weights = enforce_constraints_net7(net, weights);
            %end

            changed_signs = (grad < 0 & prev_grad > 0) | (grad > 0 & prev_grad < 0);
            gain(changed_signs) = .95 * gain(changed_signs);
            same_signs = (grad < 0 & prev_grad < 0) | (grad > 0 & prev_grad > 0);
            gain(same_signs) = gain(same_signs) + .05;
            
            %gain = (1 - .05 * (sign(grad) == sign(-prev_grad))) .* gain + ...
            %    .05 * (sign(grad) == sign(prev_grad));
            
            prev_grad = grad;
        end
        trainerr(i) = err;
        if params.adjust_rate && i > 6 && sum(diff(trainerr((i-6):(i-1))) > 0) > 2 && alphachange_steps > 5
            %alpha = alpha / 2;
            alpha = alpha * .8;
            fprintf('Decreasing learning rate to %g\n', alpha);
            alphachange_steps = 0;
        end
        if params.adjust_rate && i > 6 && alphachange_steps > 5
            prob = .3 * sum(diff(trainerr((i-6):(i-1))) < 0) / 6;
            if rand < prob
                alpha = alpha * 1.05;
                %alpha = alpha + .0001;
                alphachange_steps = 0;
                fprintf('Increasing learning rate to %g\n', alpha);
            end
        end
        alpha = alphadecay * alpha;
        
%         if alphared_steps == 20
%             alpha = alpha / 2;
%             fprintf('Reduced learning rate to %g\n', alpha);
%             alphared_steps = 0;
%         end
        
        alphachange_steps = alphachange_steps + 1;
        
        W = weightstruct_net7(net, weights);
        valout = fprop_net7(net, input.val, W, true);

        valerr(i) = -sum(sum(input.val.targets .* log(max(tiny, valout)))) / input.val.nitems;
        if(valerr(i) < best_valerr)
            best_valerr = valerr(i);
            best_weights = weights;
        end
        fprintf('%d (%g): Training error: %g, validation error: %g\n', i, alpha, trainerr(i), valerr(i));
% 
%         if i > 10 && min(valerr(i-9:i)) > best_valerr
%             alpha = alpha / 2;
%             fprintf('Adjusted learning rate to %g because of validation error\n', alpha);
%         end
        
        if params.early_stopping && i > 20 && min(valerr(i-19:i)) > best_valerr && valerr(i) - best_valerr > .1
            break
        end
        
        if mod(i, 10) == 0
            save(sprintf('nn7-%s.%d.mat', id, i));
        end
    end
    
    valerr = valerr(1:i);
    trainerr = trainerr(1:i);
end

