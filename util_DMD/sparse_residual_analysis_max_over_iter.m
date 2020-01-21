function [all_U, all_objective, all_nnz] = ...
    sparse_residual_analysis_max_over_iter(...
    my_model_base, num_iter, rank_vec, max_func)
% Wrapper for the sparse_residual_analysis function that iterates over rank
% AND sparsity, returning the best signals determined via maximum
% auto-correlation (acf) for each rank.
%   Note that rank_vec can be a scalar, so that this returns only a single
%   model
%
% Input:
%   my_model_base - a model to a) extract the data from, and b) to
%       preprocess the data so that everything is consistent
%           Note: can be just a matrix
%   num_iter (100) - number of iterations of the sparsity algorithm, i.e.
%       sequential least squares thresholding
%   rank_vec (1:25) - vector or scalar of the ranks to try, aka the number
%       of control signals allowed, aka the number of rows in the matrix U
%   max_func ('acf') - a function or metric to use for maximization. If a
%       string, assumes the pre-defined 'acf' or 'aic'
%
% Output: 
%   all_U - Cell array of U matrices across rank possibilities. Still a
%       cell array even if rank_vec is a scalar (i.e. type consistent)
%   all_objective - The autocorrelations or aic of each row of U in the 
%       same cell array format as 'all_U'
%   all_nnz - The number of nonzero entries of each row of U in the same
%       format as 'all_U'
if ~exist('num_iter', 'var') || isempty(num_iter)
    num_iter = 100;
end
if ~exist('rank_vec', 'var') || isempty(num_iter)
    rank_vec = 1:25;
end
if ~exist('max_func', 'var') || isempty(max_func)
    max_func = 'acf';
end
if ~exist('remove_isolated_spikes', 'var')
    remove_isolated_spikes = false;
end

num_ranks = length(rank_vec);
all_U = cell(num_ranks, 1);
all_objective = cell(num_ranks, 1);
all_nnz = cell(num_ranks, 1);
if isa(my_model_base, 'CElegansModel')
    m = my_model_base.dat_sz(2);
    dat = my_model_base.dat;
else
    m = size(my_model_base, 2); % Just a matrix
    dat = my_model_base;
end
% Define the function to maximize over
if ischar(max_func)
    if strcmp(max_func, 'acf')
        max_func = @(dat, U) acf(U, 1, false);
        max_func_str = 'acf';
    elseif strcmp(max_func, 'aic')
        % Use a 10-step error estimate
        max_func = @(dat, U) -aic_2step_dmdc(dat, U, [], [], 10);
        max_func_str = 'aic';
    elseif strcmp(max_func, 'false_positives')
        % Use a 10-step error estimate
        max_func = @(dat, U) -aic_2step_dmdc(dat, U, [], [], 10);
        max_func_str = 'aic';
    end
else
    assert(isa(max_func, 'function_handle'),...
        "Must pass a function handle or 'aic' or 'acf'")
end

% Build the sparse signals; plot vs. sparsity later
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
    % Choose sparsity based on max acf OR aic
    all_objective{i} = zeros(this_rank, num_iter);
    if strcmp(max_func_str, 'acf')
        % acf is calculated for each signal
        for i2 = 1:this_rank
                for i3 = 1:num_iter
                    this_U = U{i3}(i2,:)';
                    all_objective{i}(i2, i3) = acf(this_U, 1, false);
                end
            [~, which_sparsity] = max(all_objective{i}(i2, :));
            all_U{i}(i2,:) = U{which_sparsity}(i2,:);
        end
    else
        % aic and other functions are calculated for the entire matrix 'U'
        %   NOTE: aic is NEGATIVE here
        % So we need to copy them to an entire column to make the format
        % work
        for i3 = 1:num_iter
            this_U = U{i3};
            all_objective{i}(:, i3) = ...
                ones(this_rank,1)*max_func(dat, this_U);
        end
        [~, which_sparsity] = max(all_objective{i}(1, :));
        all_U{i} = U{which_sparsity};
    end
    for i3 = 1:num_iter
        all_nnz{i}(i3) = nnz(U{i3});
    end
end
end

