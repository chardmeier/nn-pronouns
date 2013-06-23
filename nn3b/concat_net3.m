function weights = concat_net3(net, W)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    weights = [W.srcembed(:); W.antembed(:); W.embhid(:); W.hidout(:)]';
end

