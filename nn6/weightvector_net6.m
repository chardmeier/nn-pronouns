function vec = weightvector_net6(W)
% Convert NN6 weights from structure to flat vector.

    vec = full([W.srcembed(:); W.antembed(:); ...
        W.linkAhid(:); W.AhidAres(:); ...
        W.embhid(:); W.hidout(:)]);
end
