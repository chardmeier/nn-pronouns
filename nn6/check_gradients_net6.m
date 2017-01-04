function Success = check_gradients_net6(net, inp, W, order)
% Check the gradients obtained by backpropagation against a
% finite differences approximations. Order is the order of the
% approximation.
    
    function S = score(inp, out)
        S = -sum(inp.targets(:) .* log(out(:)));
    end

    delta = sqrt(eps);
    
    function xx = evalf(range, W, f, idx)
        xx = zeros(2 * range, 1);
        offsets = [-range:-1 1:range] * delta;
        startx = W.(f)(idx);
        for ii = 1:(2*range)
            W.(f)(idx) = startx + offsets(ii);
            out = fprop_net6(net, inp, W, true);
            xx(ii) = score(inp, out);
        end
    end

    nchecks = 20;
    threshold = .01;
    
    %W = init_weights_net6(net, @(x,y) .3 * randn(x,y));
    [baseout, internal] = fprop_net6(net, inp, W, true);
    grads = bprop_net6(net, inp, internal, baseout, W);
    %basescore = score(inp, baseout);
    
    Success = true;
    
    %fields = fieldnames(W);
    fields = {'linkAhid', 'AhidAres'};
    for fx = 1:length(fields)
        f = fields{fx};
        fprintf('Checking %s...\n', f);
        %tocheck = randperm(numel(W.(f)), nchecks);
        tocheck = 1:20;
        for i = 1:nchecks
            idx = tocheck(i);
            fprintf('Element %d... ', idx);
            W2 = W;
            
            switch order
                case 1
                    diffgrad = [-1 1] * evalf(1, W, f, idx) / ...
                        (2*delta);
                case 2
                    diffgrad = [1 -8 8 -1] * evalf(2, W, f, idx) / ...
                        (12*delta);
                case 3
                    diffgrad = [-1 9 -45 45 -9 1] * evalf(3, W, f, idx) / ...
                        (60*delta);
                case 4
                    diffgrad = [3 -32 168 -672 672 -168 32 -3] * evalf(4, W, f, idx) / ...
                        (840*delta);
                otherwise
                    error('Order %d finite differences not implemented.', order);
            end
            
            fprintf('g = %g, d = %g: ', full(grads.(f)(idx)), diffgrad);
            if abs(grads.(f)(idx)) > 1e-10
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

