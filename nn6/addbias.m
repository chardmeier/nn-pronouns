function out = addbias(in)
% Add an initial column of ones to a matrix
    out = [ones(size(in, 1), 1), in];
end

