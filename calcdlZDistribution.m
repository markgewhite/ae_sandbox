% t-SNE code for computing pairwise joint probabilities
function [dlQ, dlN] = calcdlZDistribution( dlZ )

    % center Z
    dlZ = dlZ - mean(dlZ);

    % remove dimension labels to enable calculations below
    dlZ = stripdims(dlZ);

    % Compute joint probability that point i and j are neighbors
    sum_dlZ = sum(dlZ .^ 2);

    % Student-t distribution
    dlN = 1 ./ (1 + sum_dlZ + sum_dlZ' - 2*(dlZ')*dlZ);
    
    % set diagonal to zero
    dlN(1:size(dlZ,2)+1:end) = 0;
    
    % normalize to get probabilities
    dlQ = max(dlN ./ sum(dlN(:)), realmin);

end
