function [all_intercepts, B_lasso] = lasso_dmd(X, U,...
    which_fit, num_lambda, verbose)
% Uses MATLAB's lasso() function in a loop to calculate a sparse linear
% model
if ~exist('which_fit', 'var')
    which_fit = 8;
end
if ~exist('num_lambda', 'var')
    num_lambda = 10;
end
if ~exist('verbose', 'var')
    verbose = true;
end
all_intercepts = zeros(size(U,1),1);
B_lasso = zeros(size(U,1), size(X,1));
if verbose
    tstart = tic;
end
for i = 1:size(U,1)
    if verbose && (mod(i,10)==0 || i==1 || i==size(U,1))
        fprintf('Solving row %d/%d\n', i, size(U,1))
    end
    [all_fits, fit_info] = lasso(X', U(i,:), 'NumLambda',num_lambda);
    all_intercepts(i) = fit_info.Intercept(which_fit);
    B_lasso(i,:) = all_fits(:,which_fit); % Which fit = determined by eye
end
if verbose
    toc(tstart);
end

end

