function [gradients] = bprop_net3(net, input, internal, output, targets, weights)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    W = weights_net3(net, weights);

    % The last output of the softmax is unconnected
    output = output(:,1:end-1);
    targets = targets(:,1:end-1);

    outlayer_inputgrads = output - targets;

    %errors = 1 ./ (1 - output - targets);
    %outlayer_inputgrads = output .* (1 - output) .* errors;
    
    whidout_grads = [ones(input.nitems, 1), input.nada, internal.hidden]' * outlayer_inputgrads;
    hidlayer_inputgrads = internal.hidden .* (1 - internal.hidden) .* (outlayer_inputgrads * W.hidout(3:end,:)');
    
    wembhid_grads = [ones(input.nitems, 1), input.nada, internal.embed]' * hidlayer_inputgrads;
    emblayer_inputgrads = internal.embed .* (1 - internal.embed) .* (hidlayer_inputgrads * W.embhid(3:end,:)');
    
    wsrcemb_grads = zeros(length(net.srcvoc), net.srcembed);
    for i = 1:net.srcngsize
        for j = 1:input.ntrain
            wsrcemb_grads(input.src(j,i),:) = wsrcemb_grads(input.src(j,i),:) + emblayer_inputgrads(j,((i-1)*net.srcembed+1):i*net.srcembed);
        end
    end
    
    wantemb_grads = zeros(length(net.tgtvoc), net.antembed);
    for i = 1:size(input.ant)
        for j = 1:size(input.ant{i})
            wantemb_grads(input.ant{i}(j),:) = wantemb_grads(input.ant{i}(j),:) + ...
                emblayer_inputgrads(i,(net.srcngsize*net.srcembed + 1):end) * input.antweights{i}(j);
        end
    end
    
    if isfield(net, 'coupled_hidout')
        for i = 1:length(net.coupled_hidout)
            for j = 1:length(net.coupled_hidout{i})
                g = sum(whidout_grads(i+1, net.coupled_hidout{i}{j}));
                whidout_grads(i+1, net.coupled_hidout{i}{j}) = g;
            end
        end
    end

    wsrcemb_vec = reshape(wsrcemb_grads, 1, []);
    wantemb_vec = reshape(wantemb_grads, 1, []);
    wembhid_vec = reshape(wembhid_grads, 1, []);
    whidout_vec = reshape(whidout_grads, 1, []);
    
    gradients = [wsrcemb_vec wantemb_vec wembhid_vec whidout_vec] + net.regulariser * weights;
    if(isfield(net, 'l1regulariser'))
        gradients = gradients + net.l1regulariser * sign(weights);
    end
end

