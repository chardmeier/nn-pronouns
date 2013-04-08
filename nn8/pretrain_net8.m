function W = pretrain_net8(id, net, inp, params)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    if numel(params.nsteps) == 5
        nsteps = params.nsteps;
    else
        nsteps = params.nsteps(ones(1, 5));
    end
    
    W = weightstruct_net8(net, zeros(1, net.nweights));
    
    bmeans = ((1:net.betasensors)' - 0.5) / net.betasensors;
    bvar = 1 / net.betasensors .^ 2;
    bfactor = (bmeans .* (1 - bmeans) / bvar - 1);
    W.betasensors(:,1) = log(bmeans .* bfactor);
    W.betasensors(:,2) = log((1 - bmeans) .* bfactor);

    fprintf('Pretraining antembed...\n');
    ant = net.antwvecs(inp.ant(:,2),2:end);
    antval = net.antwvecs(inp.val.ant(:,2),2:end);
    params.nsteps = nsteps(1);
    W.antembed = train_dae([id '-antembed'], ant, antval, net.Ahid1, params);
    
    fprintf('Pretraining Ahid2...\n');
    [~,h] = fprop_net8(net, inp, W, true);
    ante = h.Ahid1agg(:,2:end);
    [~,h] = fprop_net8(net, inp.val, W, true);
    anteval = h.Ahid1agg(:,2:end);
    clear ant antval h
    params.nsteps = nsteps(2);
    W.Ahid1Ahid2 = train_dae([id '-Ahid1Ahid2'], ante, anteval, net.Ahid2, params);

    fprintf('Pretraining srcembed...\n');
    ctx = net.srcwvecs(inp.src(:),2:end);
    ctxval = net.srcwvecs(inp.val.src(:),2:end);
    params.nsteps = nsteps(3);
    W.srcembed = train_dae([id '-srcembed'], ctx, ctxval, net.srcembed, params);
    
    fprintf('Pretraining srcembjoin...\n');
    [~,h] = fprop_net8(net, inp, W, true);
    emb = h.srcembcomplete;
    [~,h] = fprop_net8(net, inp.val, W, true);
    embval = h.srcembcomplete;
    clear ctx ctxval h
    params.nsteps = nsteps(4);
    W.srcembjoin = train_dae([id '-srcjoin'], emb, embval, net.srcjoin, params);
    
    fprintf('Pretraining joinhid...\n');
    [~,h] = fprop_net8(net, inp, W, true);
    join = [inp.nada(inp.antmap,:), h.join(inp.antmap,1:net.srcjoin), h.Ahid2];
    [~,h] = fprop_net8(net, inp.val, W, true);
    joinval = [inp.val.nada(inp.val.antmap,:), h.join(inp.val.antmap,1:net.srcjoin), h.Ahid2];
    clear emb embval h
    params.nsteps = nsteps(5);
    W.joinhid = train_dae([id '-joinhid'], join, joinval, net.hidden, params);
    
    W.hidout = .001 * randn(size(W.hidout));
    save(sprintf('pretrain-%s.mat', id), 'W');
end

