function vec = weightvector_net6(W)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    vec = full([W.srcembed(:); W.antembed(:); ...
        W.linkAhid(:); W.AhidAres(:); ...
        W.embhid(:); W.hidout(:)]);
end
