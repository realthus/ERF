function [L,D]= my_constructA(X,options)
    if nargin <2
       error('parameters are not enough!');
    end

    if (~exist('options','var'))
       options = [];
    end
    
    if ~isfield(options,'k')
        options.k = 3;
    end

    if ~isfield(options,'WeightMode')
    options.WeightMode = 'binary';
    end

    [m,n] = size(X);
    W = zeros(m,m);
    dist = EuDist2(X(1:m,:),X,0);
    [val,idx] = sort(dist,2);
    idx = idx(:,1:options.k);   
    val = val(:,1:options.k);
    switch options.WeightMode
        case 'binary'       
            for i = 1:m
                W(i,idx(i,:)) = 1;
            end
        case 'heatkernel'
            if ~isfield(options,'t')
                options.t = 10;
            end
            val = exp(-val/(2*options.t^2));
            for i = 1:m
                W(i,idx(i,:)) = val(i,:);
            end
        otherwise
            error('WeightMode does not exist!');
    end  
    W = sparse(W);
    W = W - diag(diag(W));
    W = max(W,W');
    D = diag(sum(W,2));
    L = D - W;
end