% t-SNE code for computing pairwise joint probabilities
function P = calcXDistribution( X )

    doPCA = false;
    perplexity = 15;

    % Normalize input data
    X = X - min(X(:));
    X = X / max(X(:));
    X = X - mean(X,2);
    %X = (X-mean(X,2))./std(X);

    % Perform preprocessing using PCA
    if doPCA
        disp('Preprocessing data using PCA...');
        if size(X, 2) < size(X, 1)
            C = X' * X;
        else
            C = (1 / size(X, 1)) * (X * X');
        end
        [M, lambda] = eig(C);
        [lambda, ind] = sort(diag(lambda), 'descend');
        M = M(:,ind(1:initial_dims));
        lambda = lambda(1:initial_dims);
        if ~(size(X, 2) < size(X, 1))
            M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
        end
        X = bsxfun(@minus, X, mean(X, 1)) * M;
        clear M lambda ind
    end
    
    % Compute pairwise distance matrix
    sum_X = sum(X.^2);
    D = sum_X + sum_X' -2*(X')*X;
    
    D = D - min(D(:));
    D = D / max(D(:));


    % Compute joint probabilities
    P = d2pAE(D, perplexity, 1e-4);

    % Make sure P-vals are set properly
    P(1:size(P,1) + 1:end) = 0;                                 % set diagonal to zero
    P = 0.5 * (P + P');                                 % symmetrize P-values
    P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one

end