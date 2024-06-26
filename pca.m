function [U, S] = pca(X)

[m, n] = size(X);

U = zeros(n);
S = zeros(n);

Sigma = X'*X;
[U,S,V] = svd(Sigma);

end
