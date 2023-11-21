function [X_new ,Y_new , Ri_order] = RotationForest(X,Y,K)
N = size(X,1);
n = size(X,2);
M = floor(n/K);
subMethod = 1;

if(rem(n,K) ~= 0)
    K = K+1;
end

F_sub = {};

randcol = randperm(n,n);
for i = 1:K
    if(i == K)
         F_sub{i,1} = randcol(1,(1+(i-1)*M:end));
        break;
    end
    F_sub{i,1} = randcol(1,(1+(i-1)*M:i*M));
end

boostrap = 0.75;
subRi=cell(1,K);
Ri = zeros(n,n);

for i = 1:K
     if(subMethod == 1)
         subRow1 = find(Y == 1);
         subRow1 = subRow1(:,1:ceil(size(subRow1,2)*boostrap));
         subRow2 = find(Y == 2);
         subRow2 = subRow1(:,1:ceil(size(subRow2,2)*boostrap));
         Row = [subRow1 subRow2];
         randRow = randperm(size(Row,2),size(Row,2));
         Row = Row(:,randRow);
         subTrain = X(Row,F_sub{i,1}); 
     elseif(subMethod == 2)
         Row = randperm(N,round(N*boostrap));
         subTrain = X(Row,F_sub{i,1});
     end
 
     [U,~] = pca(subTrain);
     subRi{1,i} = U;
     if(i == K)
         Ri((1+(i-1)*M):end,(1+(i-1)*M):end) = subRi{1,i};
        break;
    end
     Ri((1+(i-1)*M):(i*M),(1+(i-1)*M):(i*M)) = subRi{1,i};
end

Ri = [Ri;randcol];
Ri_order = sortrows(Ri',size(Ri,1))';
Ri_order = Ri_order(1:end-1,:);

X_new = X*Ri_order;
Y_new = Y;

end