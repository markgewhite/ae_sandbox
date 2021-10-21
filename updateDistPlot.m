% *************************************
% Update the distributions plot
% *************************************

function updateDistPlot( ax, dlZ )

Z = extractdata( dlZ );
nPts = 101;
nCodes = size( Z, 1 );

hold( ax, 'off');
for i = 1:nCodes   
    pdZ = fitdist( Z(i,:)', 'Kernel', 'Kernel', 'epanechnikov' );
    ZMin = prctile( Z(i,:), 0.001 );
    ZMax = prctile( Z(i,:), 99.999 );
    ZPts = ZMin : (ZMax-ZMin)/(nPts-1) : ZMax;
    Y = pdf( pdZ, ZPts );
    Y = Y/sum(Y);
    plot( ax, ZPts, Y, 'LineWidth', 1 );
    hold( ax, 'on' );
end
hold( ax, 'off');

title( ax, 'Latent Distribution' );
xlabel( ax, 'Z' );
ylabel( ax, 'Q(Z)' );

drawnow;

end