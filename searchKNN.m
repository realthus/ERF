function  [neworder] = searchKNN(X1,X0_test)
k = 3;
[order,d] = knnsearch(X1(:,1:end-1),X0_test(:,1:end-1),'k',k);
neworder = [];

for i = order'
    X1label = X1(i,end);
    if(size(unique(X1label),1) == size(X1label,1))
        neworder = [neworder;i(1,1)];
    else
        table=tabulate(X1label);
        MaxPercent=max(table(:,3));
        [row,col]=find(table==MaxPercent);
        multiclass = table(row(1),1);
        tmp = find(X1label == multiclass);
        neworder = [neworder;i(tmp(1,1),1)];
    end
    
end

end
