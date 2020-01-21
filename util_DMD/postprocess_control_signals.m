function [U_best, all_improvement, new_U] = ...
    postprocess_control_signals(dat, all_U, all_A, all_B, s)
% Postprocessing of control signals using various metrics; default is AICc
% based. Implements a greedy algorithm to minimize the chosen metric,
% testing each individual control signal entry
%   TODO: get to work with more than one dimension

%% Set defaults
if ~exist('s','var')
    s = struct();
end
defaults = struct(...
    'verbose', 2, ...
    'min_starting_entries', 20,...
    'max_starting_entries', 200,...
    'burnin', [30, 5],...
    'start_mode', 'early',... % Simply starts at the end of the burn-in
    ...% Particularly for time-delay embedded data, should we count all rows?
    'only_use_first_row', false,...
    'use_mean_of_rows', false,...
    ...% Should we count errors below the noise level?
    'hard_threshold_noise_level', false,...
    ...% an improvement of 0 is not black and white
    'improvement_threshold', 0,...
    ...% A false kick won't change the error in the entire time series, but only
    ...% as far as the largest eigenvalue
    'calc_kick_range', false,...
    ...% Instead of comparing changes in global error and implying that each
    ...% individual kick is informed by all the data, maybe we can use a small
    ...% window to calculate local aicc values instead
    ...%   Note: this means the base error won't make sense and will have to be
    ...%   calculated for each entry separately (i.e. just keep the matrix)
    'use_local_aicc', true,...
    ...% superceded by above 'use_local_aicc'
    'err_mode', [],...
    'remove_noise_kicks', true,...
    'noise_p_thresh', 0.05,...
    ... % Estimating the noise level is very difficult!
    'noise_estimation_mode', [],...
    ...% If the data is actually time-delay embeded, this should be only
    ...% the original data rows
    'row_ind', 1:size(dat,1),...
    ...% For speed, it's good to remove some signals from the removal
    ...% rotation if they are "definitely" real
    'safe_ind_thresh', -100, ...
    ...% For speed, it's good to remove multiple signals at once if they 
    ...% are "definitely" fake
    'multiple_removal_thresh', 100,...
    'aic_mode', 'standard');
for key = fieldnames(defaults).'
    k = key{1};
    if ~isfield(s, k)
        s.(k) = defaults.(k);
    end
end

if strcmp(s.aic_mode, 'standard')
    aic_func = @(x, range_ind, range, err_cov) log(norm(...
        x(s.row_ind, range_ind), 'fro'))*range;
elseif strcmp(s.aic_mode, 'stanford')
    aic_func = @(x, range_ind, range, err_cov) norm(...
        x(s.row_ind, range_ind), 'fro')/err_cov;
else
    error('Unknown mode for calculating aic')
end

assert(s.safe_ind_thresh < s.improvement_threshold,...
    "The improvement threshold must be lower than the 'safe ind' threshold")

if isempty(s.noise_estimation_mode)
    if length(s.row_ind) < 3
        s.noise_estimation_mode = 'lillietest';
    else
        % Otherwise the lillietest is too long
        s.noise_estimation_mode = 'svd';
    end
end

%---------------------------------------------
%% Get features of the control signals in order to initialize
%---------------------------------------------
num_initial_signals = length(all_U);
all_err = zeros(num_initial_signals, 1);
all_BUnorm = all_err;
all_acf = all_err;
all_nnz = all_err;
x0 = dat(:, 1);
for i = 1:num_initial_signals
    U = all_U{i};
    tmp = calc_reconstruction_dmd(x0, [], all_A{i}, all_B{i}, U);
    all_BUnorm(i) = norm(all_B{i}*U, 'fro');
    all_nnz(i) = nnz(U);
    this_err = tmp - dat;
    all_err(i) = norm(this_err, 'fro') * ...
        (length(s.row_ind) / length(x0)); % Average over used rows
    for i2 = 1:size(U,1)
        all_acf(i) = all_acf(i) + acf(U(i2,:), 1, false) / size(U,1);
    end
end
%---------------------------------------------
%% Get an estimate for the noise level
%---------------------------------------------
if strcmp(s.noise_estimation_mode, 'lillietest')
    noise_level = estimate_noise_lillietest(dat,...
        s.noise_p_thresh, s.row_ind);
elseif strcmp(s.noise_estimation_mode, 'svd')
    [~, ~, dat_noise] = calc_snr(dat);
%     noise_level = median(median(abs(dat_noise)));
%     noise_level = mean(std(dat_noise));
%     noise_level = 2*mean(std(dat_noise));
    noise_level = norm(dat_noise, 'fro') / size(dat,2);
else
    warning('Not estimating noise level; some methods will not work')
end

%---------------------------------------------
%% Get the starting point
%---------------------------------------------
% First, try to automatically find a good starting point

if strcmp(s.start_mode, 'error')
    tmp = all_err + all_BUnorm;
    [~, min_ind] = min(tmp(s.burnin(1):end-s.burnin(2)));
    start_ind = min_ind + s.burnin(1) - 1;
elseif strcmp(s.start_mode, 'acf')
    [~, start_ind] = max(all_acf);
elseif strcmp(s.start_mode, 'early')
    start_ind = s.burnin(1);
end
U0 = all_U{start_ind};

% Catch if above logic produced a bad start
if start_ind==s.burnin(1) || start_ind==size(U0,2)-s.burnin(2) ||...
        nnz(U0)<=s.min_starting_entries || nnz(U0) <= nnz(all_U{end})
    if s.verbose >= 1
        warning('Automatic detection of starting point failed; using default')
    end
    start_ind = find(all_nnz / size(U0, 1) <=...
        s.max_starting_entries, 1);
    U0 = all_U{start_ind};
end

if s.verbose >= 1
    fprintf('Starting at control signal %d with %d entries\n',...
        start_ind, nnz(U0));
end

%---------------------------------------------
%% Settings for how to calculate the improvement
%---------------------------------------------

n = nnz(U0) - 1;
new_U = cell(n, 1);
new_U{1} = U0;
x0 = dat(:,1);
A = all_A{start_ind};
B = all_B{start_ind};

safe_ind = [];
% First remove kicks that are below the noise level
if s.remove_noise_kicks
    for i2 = 1:5
        % Note: linear indexing works even if U has more than one row
        ctr_ind = find(U0);
        to_remove = false(size(ctr_ind));
        for i = 1:length(ctr_ind)
            which_channel = ceil(ctr_ind(i)/size(U0,2));
            if max(abs(B(:, which_channel)*U0(ctr_ind(i)))) < noise_level
                to_remove(i) = true;
            end
        end
        U0(ctr_ind(to_remove)) = 0;
        [A, B] = exact_dmdc(dat, U0);
        if isempty(to_remove)
            break
        end
    end
    if s.verbose >= 1
        fprintf('Removal of noise-level kicks leaves %d entries\n', ...
            nnz(U0));
    end
    new_U{1} = U0;
    n = nnz(U0) - 1;
end
tmp = calc_reconstruction_dmd(x0, [], A, B, U0);
base_err_mat = tmp - dat;

% Other manipulations of the error matrix
if s.only_use_first_row
    base_err_mat = base_err_mat(1,:);
end
if s.use_mean_of_rows
    err_mat_factor_base = 1/length(x0);
else
    err_mat_factor_base = 1;
end
if s.hard_threshold_noise_level
    base_err_mat(base_err_mat.^2 < noise_level) = 0;
end
if strcmp(s.err_mode, 'raw')
    base_err = norm(base_err_mat, 'fro');
elseif strcmp(s.err_mode, 'BU')
    base_err = norm(base_err_mat, 'fro') + norm(B*U0, 'fro');
elseif strcmp(s.err_mode, 'aic')
    base_err = log(norm(base_err_mat, 'fro'))*size(U0, 2)*err_mat_factor_base...
        + 2*nnz(U0);
elseif strcmp(s.err_mode, 'aicc')
    B(abs(B)<1e-6) = 0;
    A(abs(A)<1e-6) = 0;
    k_t = nnz(U0) + nnz(A) + nnz(B); % Total number of parameters
    correction = (2*k_t.^2 + 2*k_t) / abs(size(U0, 2) - k_t - 1);
    if s.calc_kick_range
        % i.e. how many steps will a kick influence, above the
        % noise level (usually ~200)
        %   Note that many kicks are themselves nearly at the noise
        %   level, i.e. B*U ~ noise_level
        %   Note that this is a an approximation, averaged over all control
        %   signals
        BU = B*U0;
        BU = abs(BU(1,:));
        BU = mean(BU(BU>0));
        err_mat_factor = max( log(noise_level/BU) / ...
            log(abs(eigs(A, 1))) * err_mat_factor_base, 0);
    else
        err_mat_factor = err_mat_factor_base*size(U0, 2);
    end
    aic = log(norm(base_err_mat, 'fro'))*err_mat_factor...
        + 2*nnz(U0);
    base_err = aic + correction;
elseif strcmp(s.err_mode, 'dyn_to_ctr')
    base_err = norm(base_err_mat(1,:), 'fro') + norm(B(1)*U0, 'fro');
end

%---------------------------------------------
%% Core Iterative algorithm
%---------------------------------------------
% Throw out the signal that improves the full reconstruction the least
for i_U = 1:n
    if s.verbose >= 2
        fprintf('Iteration %d/%d...\n', i_U, n)
    end
    U = new_U{i_U};
    ctr_ind = setdiff(find(U), safe_ind);
    num_ctr = length(ctr_ind);
    all_improvement = zeros(num_ctr, 1);
    [A, B] = exact_dmdc(dat, U);
%     lambda = abs(eigs(A, 1));
%     [V, D] = eig(A);

%     parfor i_ctr = 1:num_ctr
    for i_ctr = 1:num_ctr
        % Remove the test control signal
        test_U = U;
        test_U(ctr_ind(i_ctr)) = 0;
        
        % Get dynamics matrices and reconstruction
%         [A, B] = exact_dmdc(dat, test_U);
        tmp = calc_reconstruction_dmd(x0, [], A, B, test_U);
        err_mat = tmp - dat;
        err_mat = err_mat(s.row_ind, :);
        if s.only_use_first_row
            err_mat = err_mat(1,:);
            base_err_mat = base_err_mat(1,:);
        end
        if s.hard_threshold_noise_level
            err_mat(err_mat.^2 < noise_level) = 0;
        end
        if s.use_local_aicc
            % TODO: add 2nd order correction
            [which_channel, t_ind] = ind2sub(size(U), ctr_ind(i_ctr));
%             which_channel = ceil(ctr_ind(i_ctr)/size(U0,2));
%             BU = B(s.row_ind, which_channel)*U(ctr_ind(i_ctr));
            % Estimate how many data points are actually affected by the
            % control signal
%             range = round(max( log(noise_level/max(abs(BU))) / ...
%                 log(lambda), 0));
%             this_x0 = BU + dat(:, t_ind+1);
%             opt = optimset('TolX', 1);
%             range = round( fminbnd(...
%                 abs(norm(this_x0.*V.*(D.^n), 'fro') - noise_level), ...
%                 2, 300, opt));
%             range_ind = t_ind:min(t_ind + range, size(U, 2));

            % UPDATE: Better estimate of how many data points are affected
            % by the control signal
            range = find(sum(abs(base_err_mat - err_mat)...
                > noise_level, 1), 1, 'last');
            range_ind = t_ind:range;
            
            if ~isempty(range) && range > 2
                range = range - t_ind;
                % Note: the '2' in base_err is the aic contribution
                base_err = aic_func(base_err_mat,...
                    range_ind, range, noise_level) + 2 + 4/(range-2);
                this_err = aic_func(err_mat,range_ind, range, noise_level);
%                 base_err = log(norm(...
%                     base_err_mat(s.row_ind, range_ind), 'fro'))*range + ...
%                     2 + 4/(range-2);
%                 this_err = log(norm(...
%                     err_mat(s.row_ind, range_ind), 'fro'))*range;
                all_improvement(i_ctr) = base_err - this_err;
            else
                all_improvement(i_ctr) = 100.0; % i.e. just remove it
            end
            continue
        end
        if strcmp(s.err_mode, 'raw')
            all_improvement(i_ctr) = base_err - norm(err_mat, 'fro');
        elseif strcmp(s.err_mode, 'BU')
            this_BU_magnitude = norm(B*U(ctr_ind(i_ctr)), 'fro');
            all_improvement(i_ctr) = (base_err - norm(err_mat, 'fro')) ...
                + this_BU_magnitude;
        elseif strcmp(s.err_mode, 'aic')
            aic = log(norm(err_mat, 'fro'))*size(test_U, 2)*err_mat_factor_base...
                + 2*nnz(test_U);
            all_improvement(i_ctr) = base_err - aic;
        elseif strcmp(s.err_mode, 'aicc')
            % 2nd order correction
            B(abs(B)<1e-6) = 0;
            A(abs(A)<1e-6) = 0;
            k_t = nnz(test_U) + nnz(A) + nnz(B); % Total number of parameters
            correction = (2*k_t.^2 + 2*k_t) / abs(size(U0, 2) - k_t - 1);
%             aic = log(norm(err_mat, 'fro'))*(size(test_U, 2)-nnz(test_U)) ...
%                 + 2*nnz(test_U);
            if s.calc_kick_range
                % i.e. how many steps will a kick influence, above the
                % noise level (usually ~200)
                %   Note that many kicks are themselves nearly at the noise
                %   level, i.e. B*U ~ noise_level
                BU = abs(B*U(ctr_ind(i_ctr)));
                err_mat_factor = max( log(noise_level/BU(1)) / ...
                    log(abs(eigs(A, 1))) * err_mat_factor_base, 0);
            else
                err_mat_factor = err_mat_factor_base*size(test_U, 2);
            end
            aic = log(norm(err_mat, 'fro'))*err_mat_factor ...
                + 2*nnz(test_U);
            aicc = aic + correction;
            all_improvement(i_ctr) = base_err - aicc;
        elseif strcmp(s.err_mode, 'dyn_to_ctr')
            this_quality = norm(err_mat(1,:), 'fro') + ...
                norm(B(1)*test_U, 'fro');
            all_improvement(i_ctr) = base_err - this_quality;
        end
            
    end
    % Did any removals improve the error?
    [best_improvement, removal_ind] = max(all_improvement);
    % Sanity check
    if max(all_improvement) - min(all_improvement) < 1e-6
        warning('Almost no difference between different removals')
    end
    if i_U==1
        safe_ind = all_improvement < s.safe_ind_thresh;
        safe_ind = ctr_ind(safe_ind);
        if s.verbose >= 2
            fprintf("Added %d signals to a 'do not remove' list\n",...
                length(safe_ind))
        end
        n = n - length(safe_ind);
    end
    if best_improvement > s.multiple_removal_thresh
        removal_ind = all_improvement > s.multiple_removal_thresh;
        if s.verbose >= 2
            fprintf('Removing %d signals over threshold of %.2f\n',...
                nnz(removal_ind), s.multiple_removal_thresh)
        end
        n = n - nnz(removal_ind);
    else
        if s.verbose >= 2
            fprintf('The best improvement is %.5f via removal of signal at %d\n',...
                best_improvement, ctr_ind(removal_ind))
        end
    end
    if best_improvement < s.improvement_threshold
        if ~isempty(safe_ind)
            if s.verbose >= 1
                disp('Improvement is below threshold; adding safe list back in')
            end
            n = n + length(safe_ind);
            safe_ind = [];
            new_U{i_U+1} = U; % Next iteration should be same U as this
            continue
        end
        if s.verbose >= 1
            disp('Improvement is below threshold; finishing')
        end
        U_best = U;
        break
    end
    
    % Next iteration setup: remove the proper control signal
    U(ctr_ind(removal_ind)) = 0;
    new_U{i_U+1} = U;
    if s.calc_kick_range
        %   Note that this is a an approximation, averaged over all 
        % control signals
        %   NOTE: only AICc
        [A, B] = exact_dmdc(dat, U);
        BU = B*U;
        BU = abs(BU(1,:));
        BU = mean(BU(BU>0));
        err_mat_factor = max( log(noise_level/BU) / ...
            log(abs(eigs(A, 1))) * err_mat_factor_base, 0);

        B(abs(B)<1e-6) = 0;
        A(abs(A)<1e-6) = 0;
        k_t = nnz(U) + nnz(A) + nnz(B); % Total number of parameters
        correction = (2*k_t.^2 + 2*k_t) / abs(size(U, 2) - k_t - 1);
        error('Check the err_mat usage here')
%         aic = log(norm(err_mat, 'fro'))*err_mat_factor ...
%             + 2*nnz(U);
%         base_err = aic + correction;
    else
%         base_err = base_err - best_improvement;
    end
    if s.use_local_aicc
        [A, B] = exact_dmdc(dat, U);
        tmp = calc_reconstruction_dmd(x0, [], A, B, U);
        base_err_mat = tmp - dat;
    end
end

fprintf('Final signal has %d entries\n', nnz(U_best))

end

