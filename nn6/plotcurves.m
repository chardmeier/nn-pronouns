function plotcurves(varargin)
% Convenience function to plot a number of training curves with default colours.

    styles = {'b-','r-','c-','m-','bo','ro','co','mo'};
    hold off;
    for i = 1:length(varargin)
        s = mod(i, length(styles));
        if s == 0
            s = length(styles);
        end
        plot(1:length(varargin{i}), varargin{i}, styles{mod(i, length(styles))});
        hold on;
    end
end

