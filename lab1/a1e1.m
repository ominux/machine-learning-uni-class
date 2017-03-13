fileID = fopen('data.txt','r');

formatSpec = '%f %f';

sizeA = [2, Inf];

A = fscanf(fileID,formatSpec, sizeA);
fclose(fileID);
disp(A);
