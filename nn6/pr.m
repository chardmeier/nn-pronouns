function PRF = pr(test, truth)
% Compute precision, recall and F-scores for the network's predictions.

    hardtest = (test == repmat(max(test, [], 2), 1, size(test, 2)));

    true_pos = sum(hardtest .* truth);
    ntrue = sum(truth);
    npos = sum(hardtest);
    
    true_pos = [true_pos, sum(true_pos)];
    ntrue = [ntrue, sum(ntrue)];
    npos = [npos, sum(npos)];
    
    precision = true_pos ./ npos;
    recall = true_pos ./ ntrue;
    f1 = 2 .* precision .* recall ./ (precision + recall);
    
    PRF = [precision', recall', f1'];
end

