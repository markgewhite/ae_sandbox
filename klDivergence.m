% ***********************************************************************/
% Function: Kullback-Leibler Divergence
%           q is the calculated data set
%           The reference distribution, p, is assumed to be Gaussian
% ***********************************************************************/

function kld = klDivergence( q )

qSigma = std( q );
qMu = mean( q );

kld = 0.5*( -sum(log(qSigma.^2 + 1)) ...
            + sum(qSigma.^2) ...
            + sum(qMu.^2) );

end

