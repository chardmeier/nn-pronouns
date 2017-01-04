function [LC,info] = create_lcurve(id, net, input, params, samples)
% Create learning curve by training network on an increasing number of samples.

    info(length(samples)).bve = inf; % preallocation
    LC = zeros(length(samples), 1);
    
    tiny = 1e-30;
    
    for i = 1:length(samples)
        nsmp = round(samples(i) * input.nitems);
        fprintf('Run %d: %d items\n', i, nsmp);
        sperm = randperm(input.nitems, nsmp);
        smp = create_batch_net6(input, sperm);
        smp.val = input.val;
        
        [W,te,ve,bve] = nnopt_net6(sprintf('%s.lc-%f', id, samples(i)), net, smp, params);
        
        info(i).W = W;
        info(i).te = te;
        info(i).ve = ve;
        info(i).bve = bve;
        
        [testout,testinternal] = fprop_net6(net, input.test, W, true);
        info(i).PR = pr(testout, input.test.targets);
        info(i).testout = testout;
        info(i).testinternal = testinternal;
        
        testerr = -sum(sum(input.test.targets .* log(max(tiny, testout)))) / input.test.nitems;
        fprintf('Run %d test error: %g\n', i, testerr);
        info(i).testerr = testerr;
        LC(i) = testerr;
    end
end

