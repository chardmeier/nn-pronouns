function vec = weightvector_net7(W)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    vec = full([W.srcembed(:); W.srcembjoin(:); ...
        W.linkLhid(:); W.LhidLres(:); ...
        W.antembed(:); W.Ahid1Ahid2(:); ...
        W.joinhid(:); W.hidout(:)]);
end

