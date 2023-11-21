
function [ X_src_new, X_tar_new]= LSPDDA_ch3(data_source,data_target,gamma)  
 Init_options.lambda=0.1; 
 Init_options.contribution = 0.999;
if(size(data_target,1)>1000)
    Init_options.dim = 200;  
    if(size(data_target,2)>150)
        Init_options.dim = 600; 
    end
else
    Init_options.dim = round(size(data_source,1));
end


Init_options.kernel_type='rbf'; 
Init_options.gamma=gamma;            
Init_options.T=1;
Init_options.weightmode='binary';
Init_options.mode='lpp';
Init_options.K=size(data_source,1);

%% Set options
	lambda = Init_options.lambda; 
	dim = Init_options.dim;                    
	kernel_type = Init_options.kernel_type;    
	gamma = Init_options.gamma;                

	%% Calculate
    X_src = data_source;
    X_tar = data_target;
    X = [X_src; X_tar];    
    X = X*diag(sparse(1./sqrt(sum(X.^2))));

	[m,n] = size(X');
	ns = size(X_src,1);
	nt = size(X_tar,1);   
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	M = e * e';
	M = M / norm(M,'fro');                                        
	H = eye(n)-1/(n)*ones(n,n);
    
	if strcmp(kernel_type,'primal')
		[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
		Z = A' * X;
        Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
		X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
    else                                                                    
    K = TCA_kernel(kernel_type,X',[],gamma);  
    [L1,D1] = my_constructA(X_src,Init_options);
    [L2,D2] = my_constructT(X_tar,Init_options);
    L=zeros(ns+nt);
    L(1:ns,1:ns)=L1;
    L(ns+1:ns+nt,ns+1:ns+nt)=L2;
    D = diag([diag(D1);diag(D2)]);
    [A,d] = eigs(K*M*K'+(K.*L)+lambda*eye(n),K*D*K',dim,'sm');
    eigvalue = 1./diag(d);
    cRate = cumsum(eigvalue)/sum(eigvalue);
    tmp = find(cRate>Init_options.contribution);
    finaldim = tmp(1,:);
    A = A(:,1:finaldim);
    Z = A' * K;
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
    X_src_new = Z(:,1:ns)';
    X_tar_new = Z(:,ns+1:end)';
	end

 end

 
 
 
 %%  TCA_KERNEL
 % With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013
function K = TCA_kernel(ker,X,X2,gamma)            % ÇóÄÚ¿Æ¾ØÕóÄØ
 
    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq - 2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end