% ***********************************************************************/
% Function: modelGradients second stage
% ***********************************************************************/

function [  gradEnc, gradDec ] = ...
                        modelGradientsAE2( ...
                                        dlnetEnc, ...
                                        dlnetDec,  ...
                                        dlZ, dlP, dlQ, dlN )

% calculate the KL-divergence
lossKL = klDivergence( dlZ );

%lossKL = lossKL - lossKL;

% calculate the structure fidelity loss
lossFidelity = sum( (dlP - dlQ).*dlN, 'all' );

% calculate the gradients (following igul222)
[ gradEnc, gradDec ] = dlgradient( lossKL, ...
                                   dlnetEnc.Learnables, ...
                                   dlnetDec.Learnables, ...
                                   'RetainData', true );


end