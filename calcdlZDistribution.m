% t-SNE code for computing pairwise joint probabilities
function [dlQ, dlN] = calcdlZDistribution( dlZ )

    % center Z
    dlZ = dlZ - mean(dlZ);

    % take the transpose
    dlZT = dlTranspose( dlZ );

    % Compute joint probability that point i and j are neighbors
    sum_dlZ = sum(dlZ .^ 2);
    sum_dlZT = sum(dlZT.^2,2);

    % Student-t distribution
    dlN = 1 ./ (1 + sum_dlZ + sum_dlZT - 2*stripdims(dlZT)*stripdims(dlZ));
    
    % set diagonal to zero
    dlN(1:size(dlZ,2)+1:end) = 0;
    
    % normalize to get probabilities
    dlQ = max(dlN ./ sum(dlN(:)), realmin);

end


function dlQT = dlTranspose( dlQ )
    % Calculate transpose of a dlarray
    d = size( dlQ );
    dlQT = dlarray( zeros(d(2), d(1)), 'CB' );
    for i = 1:d(1)
        for j = 1:d(2)
            dlQT(j,i) = dlQ(i,j);
        end
    end

end


function dlQSq = dlVectorSq( dlQ )
    % Calculate dlV*dlV' (transpose)
    % and preserve the dlarray
    d = size( dlQ, 2 );
    dlQSq = dlarray( zeros(d, d), 'CB' );
    for i = 1:d
        for j = 1:d
            dlQSq(i,j) = sum( dlQ(:,i).*dlQ(:,j) );
        end
    end

end


function dlQSq = dlMultiply( dlQ )
    % Calculate dlP*dlP' (transpose)
    d = size( dlQ, 2 );
    dlQT = dltranspose( dlQ );
    dlQSq = dlarray( zeros(d, d), 'CB' );
    for i = 1:d
        for j = 1:d
            dlQSq(i,j) = sum( dlQ(:,i).*dlQ(:,j) );
        end
    end

end