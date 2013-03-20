function params = default_params_dae
    params = struct();

    params.nsteps = 100;
    params.stddev = .01;
    params.momentum = 0.9;
    params.alpha = .001;
    params.alphadecay = 1;
    params.batchsize = 50;
    params.adjust_rate = true;
    params.early_stopping = true;
    params.noise_threshold = .5;
    params.l2regulariser = 1e-4;
    
    params.F1 = @(x) 1 ./ (1 + exp(-x));
    params.DF1 = @(x) x .* (1 - x);
    params.F2 = @(x) 1 ./ (1 + exp(-x));

    params.use_gpu = false;
end
