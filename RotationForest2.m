function [X_new ,Y_new , Ri_order] = RotationForest2(X,Y,K)
N = size(X,1);
n = size(X,2);
M = floor(n/K);
classnum = size(unique(Y),1);
class = unique(Y)';
subMethod = 3;

if(rem(n,M) ~= 0)
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
         subRow = [];
         subClassNum = ceil(classnum/2);
         subClass = class(:,randperm(classnum,subClassNum));
         for ic = subClass
             subRow1 = find(Y == ic);
             subRow1 = subRow1(:,1:ceil(size(subRow1,2)*boostrap));
             subRow = [subRow subRow1];
         end
         randRow = randperm(size(subRow,2),size(subRow,2));
         Row = subRow(:,randRow);
         subTrain = X(Row,F_sub{i,1}); 
     elseif(subMethod == 2)
         Row = randperm(N,round(N*boostrap));
         subTrain = X(Row,F_sub{i,1});
     elseif(subMethod == 3)
         subRow = [];
         subClassNum = ceil(classnum/2);
         subClass = class(:,randperm(classnum,subClassNum));
         for ic = subClass
             subRow1 = find(Y == ic);
             subRow1 = subRow1(randperm(size(subRow1,1),ceil(size(subRow1,1)*boostrap)),:);
             subRow = [subRow;subRow1];
         end
         randRow = randperm(size(subRow,1),size(subRow,1));
         Row = subRow(randRow,:)';
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