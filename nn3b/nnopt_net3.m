function [best_weights, trainerr, valerr, best_valerr] = nnopt_net3(net, input, targets, valtgt, varargin)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    if isempty(varargin)
        stddev = .1;
        nsteps = 500;
        alpha = .001;
        alphadecay = 1;
        momentum = .9;
        batchsize = 1000;
        adjust_rate = true;
        early_stopping = true;
    else
        stddev = .1;
        momentum = .9;
        params = varargin{1};
        nsteps = params.nsteps;
        alpha = params.initialrate;
        alphadecay = params.ratedecay;
        adjust_rate = params.adjust_rate;
        batchsize = params.batchsize;
        early_stopping = params.early_stopping;
    end
    
    targets = softmax_targets(targets);
    valtgt = softmax_targets(valtgt);

    if mod(input.ntrain, batchsize) > 0
        npad = batchsize - mod(input.ntrain, batchsize);
    else
        npad = 0;
    end
    nbatch = (input.ntrain + npad) / batchsize;
    
    trainerr = zeros(1, nsteps);
    valerr = zeros(1, nsteps);
    
    if isfield(net, 'initweights')
        weights = net.initweights;
    else
        weights = stddev * randn(1, net.nweights);
    end

    if isfield(net, 'coupled_hidout')
        W = weights_net3(net, weights);
        for i = 1:length(net.coupled_hidout)
            for j = 1:length(net.coupled_hidout{i})
                w = sum(W.hidout(i+1, net.coupled_hidout{i}{j}));
                W.hidout(i+1, net.coupled_hidout{i}{j}) = w;
            end
        end
        weights = concat_net3(net, W);
    end
    
    weight_change = zeros(1, net.nweights);
    gain = ones(1, net.nweights);
    prev_grad = zeros(1, net.nweights);
    rms = zeros(1, net.nweights);
    best_valerr = inf;
    alphachange_steps = 0;
    tiny = 1e-30;
    for i = 1:nsteps
        %momentum = 1 - 1/i;
        
        batchperm = [randperm(input.ntrain), randperm(input.ntrain)];
        batchperm = reshape(batchperm(1:(input.ntrain + npad)), nbatch, batchsize);
        
        err = 0;
        cmp_trainerr = 0;
        for j = 1:nbatch
            [batchinput,batchtargets] = create_batch_net3(input, targets, batchperm(j,:));
            [output,hidden] = fprop_net3(net, batchinput, weights, false);
            grad = bprop_net3(net, batchinput, hidden, output, batchtargets, weights);
            cmp_trainerr = cmp_trainerr - sum(sum(log(abs(ones(size(output)) - output - batchtargets)))) / input.ntrain;
            err = err - sum(sum(batchtargets .* log(max(tiny, output)))) / input.ntrain;
            
            rms = 0.9 * rms + 0.1 * grad .^ 2;
            grad = grad ./ (sqrt(rms) + tiny);
            
            weight_change = momentum * weight_change - alpha * gain .* grad;
            weights = weights + weight_change;
            
            if isfield(net, 'weightconstraint')
                weights = enforce_constraints_net3(net, weights);
            end

            same_sign = (sign(grad) == sign(prev_grad));
            gain(same_sign) = min(100, gain(same_sign) + .05);

            flip_sign = (sign(grad) == sign(-prev_grad));
            gain(flip_sign) = max(.01, .95 * gain(flip_sign));
            
            prev_grad = grad;
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
        
%         if alphared_steps == 20
%             alpha = alpha / 2;
%             fprintf('Reduced learning rate to %g\n', alpha);
%             alphared_steps = 0;
%         end
        
        alphachange_steps = alphachange_steps + 1;
        
        valout = fprop_net3(net, input.val, weights, true);
        cmp_valerr = -sum(sum(log(abs(ones(size(valout)) - valout - valtgt)))) / input.nval;
        valerr(i) = -sum(sum(valtgt .* log(max(tiny, valout)))) / input.nval;
        if(valerr(i) < best_valerr)
            best_valerr = valerr(i);
            best_weights = weights;
        end
        fprintf('%d: Training error: %g (%g), validation error: %g (%g)\n', i, trainerr(i), cmp_trainerr, valerr(i), cmp_valerr);
% 
%         if i > 10 && min(valerr(i-9:i)) > best_valerr
%             alpha = alpha / 2;
%             fprintf('Adjusted learning rate to %g because of validation error\n', alpha);
%         end
        
        if early_stopping && i > 20 && min(valerr(i-19:i)) > best_valerr && valerr(i) - best_valerr > .1
            break
        end
    end
    
    valerr = valerr(1:i);
    trainerr = trainerr(1:i);
end

