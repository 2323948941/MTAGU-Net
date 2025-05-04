function XX = RMSE(Y_YC1,Y1)
    [n1,n2,n3,n4]=size(Y_YC1);
    z=n1*n2*n3*n4;
    Y_YC=reshape(Y_YC1,[z,1]);
    Y=reshape(Y1,[z,1]);
    XX=(sqrt(mean((Y_YC-Y).^2)));
end