% t-SNE code for computing pairwise joint probabilities
function [Q, num] = calcZDistribution( Z )

    % transpose Z
    Z = Z';

    % center Z
    Z = Z - mean(Z,2);

    % Compute joint probability that point i and j are neighbors
    sum_Z = sum(Z .^ 2, 2);

    % Student-t distribution
    num = 1 ./ (1 + bsxfun(@plus, sum_Z, bsxfun(@plus, sum_Z', -2 * (Z * Z'))));
    
    % set diagonal to zero
    num(1:size(Z,1)+1:end) = 0;
    
    % normalize to get probabilities
    Q = max(num ./ sum(num(:)), realmin);

end
