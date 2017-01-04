function s = sigmoid(x)
% Logistic sigmoid function.

    s = 1 ./ (1 + exp(-x));
end

