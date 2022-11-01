% ***********************************************************************/
% Function: modelGradients for an autoencoder (no adversarial design)
% ***********************************************************************/

function [  gradEnc, gradDec, lossFidelity, dlZ ] = ...
                        modelGradientsAE( ...
                                        dlnetEnc, ...
                                        dlnetDec, ...
                                        dlX, ...
                                        dlP, ...
                                        fullCalc )

% predict the fake latent code for the image
dlZ = forward( dlnetEnc, dlX );

% predict the fake image from the fake code
dlXHat = forward( dlnetDec, dlZ );

% calculate the reconstruction loss
lossRecon = mean( mean( 0.5*(dlXHat - dlX).^2, 1 ) );

if fullCalc

    % calculate the KL-divergence
    %lossKL = klDivergence( dlZ );

    % calculate the fidelity loss
    [dlQ, dlN] = calcdlZDistribution( dlZ );
    [Q, N] = calcZDistribution( extractdata(dlZ) );

    lossFidelity = sum( (dlP - dlQ).*dlN, 'all' );

else

    lossKL = 0;
end

% combine 
loss = 0.9*lossRecon + lossFidelity;

% calculate the gradients (following igul222)
[ gradEnc, gradDec ] = dlgradient( lossFidelity, ...
                                   dlnetEnc.Learnables, ...
                                   dlnetDec.Learnables, ...
                                   'RetainData', true );

end