function weights = train_rbm(id, data, val, nhid)
% Train restricted Boltzmann machine. Not used for NN6 training.

    function S = sample_bernoulli(P)
        S = rand(size(P)) < P;
    end

    alpha = .0005;
    %alpha = 1e-5;
    stddev = .01;
    batchsize = 50;
    nsteps = 10;
    momentum = .9;
    
    ndata = size(data, 1);
    nvis = size(data, 2);
    
    batches_per_epoch = ceil(ndata / batchsize);
    
    tiny = 1e-30;
    
    weights = stddev * randn(nvis + 1, nhid + 1);
    datap = mean(data, 1);
    weights(:,1) = [0; log(datap ./ (1 - datap) + tiny)'];
    
    bias = ones(batchsize, 1);
    weight_change = zeros(size(weights));
    rms = ones(size(weights));
    %gain = ones(size(weights));
    
    nvalsub = min(size(val, 1), 100);
    testsubset = data(1:nvalsub,:);
    valsub = val(1:nvalsub,:);
    
    nupdates = 0;
    for i = 1:nsteps
        fprintf('Epoch %d\n', i);
        
        for j = 1:batches_per_epoch
            batchperm = randperm(ndata, batchsize);
            batch = data(batchperm,:);

            %batch = (rand(size(batch)) < 0.5) .* batch;

            phid1 = sigmoid([bias, batch] * weights(:,2:end));
            hid1 = sample_bernoulli(phid1);
            vis2 = sigmoid([bias, hid1] * weights(2:end,:)');
            hid2 = sigmoid([bias, vis2] * weights(:,2:end));

            grads = ([bias,batch]' * [bias,phid1] - [bias,vis2]' * [bias,hid2]) / batchsize;
            grads(2:end,2:end) = grads(2:end,2:end) - .0001 * weights(2:end,2:end);
            
            rms = 0.9 * rms + 0.1 * grads .^ 2;
            grads = grads ./ sqrt(rms + tiny);
            
            %a = sum(abs(weight_change(:)));
            %b = sum(abs(weights(:)));
            %fprintf('Update: %g/%g = %g\n', a, b, a/b);
            
            weight_change = momentum * weight_change + alpha * grads;
            weights = weights + weight_change;
            
            nupdates = nupdates + 1;
            
            fprintf('.');
            if mod(nupdates, 50) == 0
                fprintf('%d\n', nupdates);
            end
            
            if mod(nupdates, 300) == 0
                hid1 = sample_bernoulli(sigmoid(addbias(valsub) * weights(:,2:end)));
                vis2 = sigmoid(addbias(hid1) * weights(2:end,:)');
                err = sum(sum((valsub - vis2) .^ 2)) / ndata;
                
                figure(1), clf, hist(weights(:),30);
                figure(2), clf, hist(weight_change(:),30);
                Ftest = -sum(testsubset * weights(2:end,1)) - ...
                    sum(sum(log1p(exp(addbias(testsubset) * weights(:,2:end)))));
                Fval = -sum(valsub * weights(2:end,1)) - ...
                    sum(sum(log1p(exp(addbias(valsub) * weights(:,2:end)))));
                
                fprintf('%d.%d: err = %g, |grad| = %f, |W| = %f, gap = %g\n', ...
                    i, j, err, norm(grads, 'fro'), norm(weights, 'fro'), Ftest - Fval);
            end
        end
        save(sprintf('rbm-%s.%d.mat', id, i));
    end
end

