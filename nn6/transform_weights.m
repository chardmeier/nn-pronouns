function out = transform_weights(fn, varargin)
% Apply a transformation function to all weights.
    out = fn(varargin{:});
end

