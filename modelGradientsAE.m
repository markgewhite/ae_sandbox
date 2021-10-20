% ***********************************************************************/
% Function: modelGradients for an autoencoder (no adversarial design)
% ***********************************************************************/

function [  gradEnc, gradDec, lossRecon ] = ...
                        modelGradientsAE( ...
                                        dlnetEnc, ...
                                        dlnetDec, ...
                                        dlX )

% predict the fake latent code for the image
dlZ = forward( dlnetEnc, dlX );

% predict the fake image from the fake code
dlXHat = forward( dlnetDec, dlZ );

% calculate the reconstruction loss
lossRecon = mean( mean( 0.5*(dlXHat - dlX).^2, 1 ) );

% calculate the KL-divergence
lossKL = klDivergence( dlZ );

% combine 
loss = 0.9*lossRecon + 0.1*lossKL;

% calculate the gradients (following igul222)
[ gradEnc, gradDec ] = dlgradient( loss, ...
                                   dlnetEnc.Learnables, ...
                                   dlnetDec.Learnables, ...
                                   'RetainData', true );

end