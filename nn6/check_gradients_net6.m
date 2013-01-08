function Success = check_gradients_net6(net, inp, W)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    function S = score(inp, out)
        S = -sum(inp.targets(:) .* log(out(:)));
    end
    
    nchecks = 10;
    delta = sqrt(eps);
    threshold = .01;
    
    %W = init_weights_net6(net, @(x,y) .3 * randn(x,y));
    [baseout, internal] = fprop_net6(net, inp, W, true);
    grads = bprop_net6(net, inp, internal, baseout, W);
    %basescore = score(inp, baseout);
    
    Success = true;
    
    fields = fieldnames(W);
    %fields = {'hidout'};
    for fx = 1:length(fields)
        f = fields{fx};
        fprintf('Checking %s...\n', f);
        tocheck = randperm(numel(W.(f)), nchecks);
        %tocheck = 225524;
        for i = 1:nchecks
            idx = tocheck(i);
            fprintf('Element %d... ', idx);
            W2 = W;
            
            W2.(f)(idx) = W.(f)(idx) - delta;
            deltaout1 = fprop_net6(net, inp, W2, true);
            deltascore1 = score(inp, deltaout1);
            W2.(f)(idx) = W.(f)(idx) + delta;
            deltaout2 = fprop_net6(net, inp, W2, true);
            deltascore2 = score(inp, deltaout2);
            
            diffgrad = (deltascore2 - deltascore1) / (2*delta);
            fprintf('g = %g, d = %g: ', full(grads.(f)(idx)), diffgrad);
            if grads.(f)(idx) > 1e-10
                rel = abs((diffgrad - grads.(f)(idx)) / grads.(f)(idx));
            else
                rel = abs(diffgrad - grads.(f)(idx));
            end
            if rel < threshold
                fprintf('ok (rel = %f)\n', rel);
            else
                fprintf('BAD (rel = %f)\n', rel);
                Success = false;
            end
        end
    end

end

