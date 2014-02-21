function vec = weightvector_deplm1(W)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    vec = full([W.srcembed(:); W.tgtembed(:); ...
        W.embhid(:); W.hidout(:)]);
end
