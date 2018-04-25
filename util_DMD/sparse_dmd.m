function [ A_sparse, all_errors ] = sparse_dmd( X, prams)
% Do sparse DMD

%% Set defaults
defaults = struct(...
    'min_tol', 1e-8,...
    'error_norm', @(A, x, b) norm(A*x-b),...
    'max_error_mult', 2,...
    'column_mode', false,...
    'rows_to_predict', size(X,1),...
    'sparsity_goal', 0.5,...
    'max_iter', 10,...
    'verbose', true);
fnames = fieldnames(prams);
for key = fieldnames(defaults).'
    k = key{1};
    if ~ismember(k,fnames)
        prams.(k) = defaults.(k);
    end
end

if isnumeric(X)
    X1 = X(:,1:end-1);
    X2 = X(:,2:end);
elseif iscell(X)
    X1 = X{1};
    X2 = X{2};
else
    error('Unrecognized data format')
end
%==========================================================================

%% Set up the convex optimization problem and solve with cvx
% Initialize with MATLAB backslash
A_sparse = X2 / X1;
A_sparse_old = A_sparse;
n = length(A_sparse); %#ok<NASGU>
% Use cvx to set up a sequential thresholding loop
total_num_elem = numel(A_sparse(1:prams.rows_to_predict,:));
num_nnz = zeros(prams.max_iter,1);
num_nnz(1) = nnz(A_sparse(1:prams.rows_to_predict,:));
all_errors = zeros(prams.max_iter,1);
all_errors(1) = prams.error_func(A_sparse, X1, X2);
error_max = all_errors(1)*prams.max_error_mult;
did_it_abort_early = false;
for i=2:prams.max_iter
    % Threshold to enforce sparsity
    sparsity_pattern = abs(A_sparse)<prams.min_tol;
    
    num_nnz(i) = nnz(A_sparse(1:prams.rows_to_predict,:));
    all_errors(i) = prams.error_func(A_sparse, X1, X2);
    if prams.verbose
        fprintf('Iteration %d:\n', i-1)
        fprintf('  %d nonzero-entries (goal: %d)\n',...
            num_nnz(i), round(total_num_elem*prams.sparsity_goal))
        fprintf('  Current error: %.4f (max=%.4f)\n',...
            all_errors(i), error_max)
    end
    % Check convergence etc
    if i>2
        current_sparsity = num_nnz(i) / total_num_elem;
        if all_errors(i) > error_max
            % If we have already halved the threshold, abort early instead
            % of bouncing around the threshold
            A_sparse = A_sparse_old;
            if length(find(diff(all_errors>error_max))) < 2
                % Note: diff() because we want to continue if:
                %   all_errors>error_max = [0 0 0 1 1 1]
                tol_factor = abs(all_errors(i)-error_max) / ...
                    (all_errors(i)+error_max);
                fprintf('  Error exceeded max; multiplying threshold by %.2f\n',...
                    tol_factor)
                prams.min_tol = prams.min_tol * tol_factor;
            else
                disp('  Error exceeded maximum again; aborting')
                did_it_abort_early = true;
                break
            end
        elseif  current_sparsity < prams.sparsity_goal
            fprintf('Achieved sparsity goal (%f percent)\n',current_sparsity)
            break
        elseif (num_nnz(i-1)-num_nnz(i)) < max([0.02*num_nnz(i), 5])
            % Measure what would be needed to get another 50% to the
            % sparseness goal and set that as the threshold
            nnz_goal = round(total_num_elem*prams.sparsity_goal);
            f = @(x) abs(length(A_sparse(A_sparse>x))-nnz_goal);
            goal_tol = fminbnd(f, prams.min_tol, max(max(A_sparse)));
            tol_factor = abs(all_errors(i)-error_max) / ...
                (all_errors(i)+error_max);
            % Move a percentaget towards the ideal goal in relation to how
            % much error we can increase
            prams.min_tol = prams.min_tol + tol_factor*goal_tol;
            
            fprintf('  Stall predicted; new tolerance is %.2f\n',...
                prams.min_tol)
        else
            A_sparse_old = A_sparse;
        end
        sparsity_pattern = abs(A_sparse)<prams.min_tol;
    else
        A_sparse(sparsity_pattern) = 0;
    end
    % Actually solve, either all at once or row by row
    if ~prams.column_mode
        cvx_begin quiet
            variable A_sparse(n,n)
            minimize( ...
                norm(A_sparse*X1-X2,2) )
            A_sparse(sparsity_pattern) == 0 %#ok<NOPRT,EQEFF>
        cvx_end
    else
        for i2 = 1:prams.rows_to_predict
            if prams.verbose && (i2==1 || mod(i2,20)==0)
                fprintf('Solving row %d/%d...\n',i2,prams.rows_to_predict)
            end
            X2_row = X2(i2,:);
            sparsity_pattern_row = sparsity_pattern(i2,:);
            
            cvx_begin quiet
                variable A_sparse_row(1,n)
                minimize( ...
                    norm(A_sparse_row*X1-X2_row,2) )
                A_sparse_row(sparsity_pattern_row) == 0 %#ok<NOPRT,EQEFF>
            cvx_end
            
            A_sparse(i2,:) = A_sparse_row;
        end
    end
end

if prams.verbose
    if did_it_abort_early
        final_index = i-1;
    else
        final_index = i;
    end
    fprintf('Number of nonzero entries decreased from %d to %d\n',...
        num_nnz(1), num_nnz(final_index));
    fprintf('Error increased from %.4f to %.4f\n',...
        all_errors(1), all_errors(final_index));
end

end

