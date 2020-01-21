function [U] = combine_signals(all_U, weights)
% Combines signals according to a weight vector
if sum(weights) ~= 1
    weights = weights / sum(weights);
end
if iscell(all_U)
    all_U = cell2mat(all_U);
end

sz = size(all_U);
U = zeros(1, sz(2));
for i = 1:sz(1)
    U = U + weights(i) * all_U(i, :);
end

end