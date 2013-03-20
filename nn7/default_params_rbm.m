function params = default_params_rbm
    params = struct();

    params.alpha = .0005;
    %alpha = 1e-5;
    params.stddev = .01;
    params.batchsize = 20;
    params.nsteps = 4;
    params.momentum = .9;
    params.l2regulariser = .001;
    
    params.sparsity_target = 0.1;
    params.sparsity_decay = 0.95;
    params.sparsity_cost = .001;

    params.use_gpu = false;
end
