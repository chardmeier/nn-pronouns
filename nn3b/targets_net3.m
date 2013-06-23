function [targets,valtargets,testtargets] = targets_net3(net, input)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    targets = zeros(input.ntrain, net.output);
    valtargets = zeros(input.nval, net.output);
    testtargets = zeros(input.ntest, net.output);
    for i = 1:length(net.targetpronouns)
        for j = 1:length(net.targetpronouns{i})
            targets(:,i) = max(targets(:,i), ...
                cellfun(@(x) max(strcmpi(net.targetpronouns{i}{j}, x)), input.anaphwords));
            valtargets(:,i) = max(valtargets(:,i), ...
                cellfun(@(x) max(strcmpi(net.targetpronouns{i}{j}, x)), input.val.anaphwords));
            testtargets(:,i) = max(testtargets(:,i), ...
                cellfun(@(x) max(strcmpi(net.targetpronouns{i}{j}, x)), input.test.anaphwords));
        end
    end
end

