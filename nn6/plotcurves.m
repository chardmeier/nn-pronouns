function plotcurves(varargin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
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

