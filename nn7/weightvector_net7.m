function vec = weightvector_net7(W)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    vec = [W.srcembed(:), W.srcembjoin(:), ...
        W.antembed(:), W.Ahid1Ahid2(:), ...
        W.linkLhid(:), W.LhidLres(:), ...
        W.joinhid(:), W.hidout(:)];
end

