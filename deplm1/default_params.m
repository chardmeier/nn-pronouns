function params = default_params()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    params = struct('nsteps', 10, ...
        'stddev', .01, ...
        'momentum', .9, ...
        'initialrate', .005, ...
        'ratedecay', 1, ...
        'adjust_rate', true, ...
        'batchsize', 500, ...
        'early_stopping', false);

end

