function Y = sampleCluster(X,N)

[Idx1,C1]=kmeans(X(:,1:end-1),N,'MaxIter',10000);  
ctLb = [];

for i = 1:N   
    t1 = find(Idx1 == i);
    t2 = X (t1, :);                       
    t3 = t2 (:, end);                   
    t4 = mean(t3);
    ctLb = [ctLb; t4];   
end

Y = [C1, ctLb];
end



    