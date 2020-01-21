function [all_U, all_acf, all_nnz] = ...
    sparse_residual_analysis_max_acf(my_model_base, num_iter, rank_vec)
% Wrapper for the sparse_residual_analysis function that iterates over rank
% AND sparsity, returning the best signals determined via maximum
% auto-correlation (acf) for each rank.
%   Note that rank_vec can be a scalar, so that this returns only a single
%   model
%
% Input:
%   my_model_base - a model to a) extract the data from, and b) to
%       preprocess the data so that everything is consistent
%   num_iter (100) - number of iterations of the sparsity algorithm, i.e.
%       sequential least squares thresholding
%   rank_vec (1:25) - vector or scalar of the ranks to try, aka the number
%       of control signals allowed, aka the number of rows in the matrix U
%
% Output: 
%   all_U - Cell array of U matrices across rank possibilities. Still a
%       cell array even if rank_vec is a scalar
%   all_acf - The autocorrelations of each row of U in the same cell array
%       format as 'all_U'
%   all_nnz - The number of nonzero entries of each row of U in the same
%       format as 'all_U'
if ~exist('num_iter', 'var')
    num_iter = 100;
end
if ~exist('rank_vec', 'var')
    rank_vec = 1:25;
end
num_ranks = length(rank_vec);
all_U = cell(num_ranks, 1);
all_acf = cell(num_ranks, 1);
all_nnz = cell(num_ranks, 1);
if isa(my_model_base, 'CElegansModel')
    m = my_model_base.dat_sz(2);
else
    m = size(my_model_base, 2); % Just a matrix
end

% Build the sparse signals; plot vs. sparsity LATER
settings = struct('num_iter', num_iter);
for i = 1:length(rank_vec)
    if length(rank_vec)>1
        this_rank = rank_vec(i);
    else
        this_rank = rank_vec;
    end 
    settings.r_ctr = this_rank;
    U = sparse_residual_analysis(my_model_base, settings);

    all_U{i} = zeros(this_rank, m-1);
    % Choose sparsity based on max acf
    all_acf{i} = zeros(this_rank, num_iter);
    for i2 = 1:this_rank
        for i3 = 1:num_iter
            dat = U{i3}(i2,:)';
            all_acf{i}(i2, i3) = acf(dat, 1, false);
        end
        [~, which_sparsity] = max(all_acf{i}(i2, :));
        all_U{i}(i2,:) = U{which_sparsity}(i2,:);
    end
    for i3 = 1:num_iter
        all_nnz{i}(i3) = nnz(U{i3});
    end
end

end

