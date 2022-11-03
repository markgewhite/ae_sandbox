% *************************************
% Update the progress plots
% *************************************

function updateProgressAE(  ax1, ...
                            lineRecon, lineAux, ...
                            reconLoss, auxLoss, ...
                            epoch, j, t0 )

addpoints( lineRecon, j, ...
            double(extractdata(reconLoss)) );

addpoints( lineAux, j, auxLoss );
        
% Update the title with training progress information.
D = duration( 0, 0, toc(t0), 'Format', 'hh:mm:ss' );
title( ax1, ...
    "Epoch: " + epoch + ", " + ...
    "Iteration: " + j + ", " + ...
    "Elapsed: " + string(D))
        

drawnow;

end