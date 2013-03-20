function [out,scaling] = scaledata(varargin)
    if nargin < 1 || nargin > 2
       error('Wrong number of input arguments for scaledata.');
    end
    in = varargin{1};

    if nargin == 2
        scaling = varargin{2};
    else
        scaling = struct();
        scaling.min = min(in, [], 1);
        scaling.max = max(in, [], 1);
    end

    out = bsxfun(@times, bsxfun(@minus, in, scaling.min), ...
        1 ./ max(1e-30, scaling.max - scaling.min));
end
