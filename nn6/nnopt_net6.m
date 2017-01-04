function [best_weights, trainerr, valerr, best_valerr] = nnopt_net6(id, net, input, params)
% Train network with a variant of the rmsprop algorithm.

    stddev = params.stddev;
    momentum = params.momentum;
    nsteps = params.nsteps;
    alpha = params.initialrate;
    alphadecay = params.ratedecay;
    adjust_rate = params.adjust_rate;
    batchsize = params.batchsize;
    early_stopping = params.early_stopping;


    if mod(input.nitems, batchsize) > 0
        npad = batchsize - mod(input.nitems, batchsize);
    else
        npad = 0;
    end
    nbatch = (input.nitems + npad) / batchsize;
    
    trainerr = zeros(1, nsteps);
    valerr = zeros(1, nsteps);
    
    if isfield(net, 'initweights')
        W = net.initweights;
    else
        W = init_weights_net6(net, @(x,y) stddev * randn(x,y));
    end
    
    if isfield(params, 'initgain')
        gain = params.initgain;
    else
        gain = init_weights_net6(net, @ones);
    end
        
    weight_change = init_weights_net6(net, @zeros);
    prev_grad = init_weights_net6(net, @zeros);
    rms = init_weights_net6(net, @ones);
    best_valerr = inf;
    alphachange_steps = 0;
    tiny = 1e-30;
    mverr = zeros(1, nsteps * nbatch);
    mv_t = 1;
    indices = [1:input.nitems, zeros(1, npad)];
    for i = 1:nsteps
        batchperm = reshape(indices(randperm(numel(indices))), nbatch, batchsize);
        
        err = 0;
        for j = 1:nbatch
            thisperm = batchperm(j, batchperm(j,:) > 0);
            batchinput = create_batch_net6(input, thisperm);
            Wstruct = weightstruct_net6(net, W);
            [output,hidden] = fprop_net6(net, batchinput, Wstruct, false);
            gradstruct = bprop_net6(net, batchinput, hidden, output, Wstruct);
            grad = weightvector_net6(gradstruct);
            W = weightvector_net6(Wstruct);

            if isfield(net, 'regulariser') && net.regulariser > 0
                grad = transform_weights(@(x,y) x + net.regulariser * y, grad, W);
            end

            if isfield(net, 'l1regulariser') && net.l1regulariser > 0
                grad = transform_weights(@(x,y) x + net.l1regulariser * sign(y), grad, W);
            end
    
            err = err - sum(sum(batchinput.targets .* log(max(tiny, output)))) / input.nitems;
            mv_t = mv_t + 1;
            mverr(mv_t) = mverr(mv_t-1) - (2/((i-1)*nbatch+j)) * (mverr(mv_t-1) - err);
            
            rms = transform_weights(@(r,g) 0.9 * r + 0.1 * g .^ 2, rms, grad);
            grad = transform_weights(@(r,g) g ./ (sqrt(r + tiny)), rms, grad);
            
            weight_change = transform_weights(@(wc, gn, gr) momentum * wc - alpha * gn .* gr, ...
                weight_change, gain, grad);
            W = transform_weights(@plus, W, weight_change);
            
            if isfield(net, 'weightconstraint')
                weights = enforce_constraints_net6(net, weights);
            end

            gain = transform_weights(@(gn,gr,pg) (1 - .05 * (sign(gr)==sign(-pg))) .* gn + ...
                (sign(gr)==sign(pg)) * .05, gain, grad, prev_grad);
            
            prev_grad = grad;
        end
        %plot(1:length(mverr), mverr, 'erasemode', 'background');
        %drawnow;
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
        
        Wstruct = weightstruct_net6(net, W);
        valout = fprop_net6(net, input.val, Wstruct, true);

        valerr(i) = -sum(sum(input.val.targets .* log(max(tiny, valout)))) / input.val.nitems;
        if(valerr(i) < best_valerr)
            best_valerr = valerr(i);
            best_weights = W;
        end
        fprintf('%d (%g): Training error: %g, validation error: %g\n', i, alpha, trainerr(i), valerr(i));
% 
%         if i > 10 && min(valerr(i-9:i)) > best_valerr
%             alpha = alpha / 2;
%             fprintf('Adjusted learning rate to %g because of validation error\n', alpha);
%         end
        
        if early_stopping && i > 20 && min(valerr(i-19:i)) > best_valerr && valerr(i) - best_valerr > .1
            break
        end
        
        if mod(i, 5) == 0
            %save('-v7.3', sprintf('nn6-%s.%d.mat', id, i));
            save('-v7.3', sprintf('nn6-%s.mat', id));
        end
    end
    
    valerr = valerr(1:i);
    trainerr = trainerr(1:i);
end

