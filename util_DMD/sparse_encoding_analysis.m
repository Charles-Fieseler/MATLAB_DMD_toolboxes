function [B_prime_lasso_td_3d, all_intercepts_td, elimination_neurons, ...
    eliminated_names, which_single_model,...
    all_err, all_fp, all_fn, num_spikes] = ...
    sparse_encoding_analysis(file_or_obj, s)
%% Sparse Encoding Analysis
% Analyzes a dataset and a (nominally) encoded signal, determining where
% and at what timescale the signal is encoded and how well, according to
% two main metrics:
%   False detection, i.e. false negatives and positives
%   Error, as determined by L2 norm or correlation
%
% Input:
%   file_or_obj - the file with the data, or the CElegansModel that
%       contains the data
%   s - the settings for this algorithm
%
% Output:
%   B_prime_lasso_td_3d - A 3d matrix of the encoding matrix, B'. Each
%       dimension encodes the following: 1, 2 are the sparse LASSO models,
%       and 3 is the sequential models as neurons are removed.
%   all_intercepts_td - The intercepts in the sparse models (constant term)
%   elimination_neurons - The indices of the eliminated neurons
%   eliminated_names - The names of the eliminated neurons
%   which_single_model - According to the field 'which_single_signal' in
%       the settings ('s' in the input), the elimination path for only a
%       single model will be calculated
%   all_err - The L2 errors of the sparse models
%   all_fp - The False Positives of the sparse models
%   all_fn - The False Negatives of the sparse models
%   num_spikes - The number of spikes in the signals; can be used to
%       normalize 'all_fp' and 'all_fn' above

%---------------------------------------------
%% Set up defaults
%---------------------------------------------
if ~exist('s','var')
    s = struct();
end
defaults = struct(...
    'verbose', true, ...
    'max_iter', 20,...
    'to_use_LASSO', true,...
    'which_fit', 4, ...
    'to_fit_all_models', false,...
    'which_single_signal', {{'DT', 'Dorsal turn'}},...
    'to_calculate_false_detection', true,...
    'which_error_metric', '',...
    'seed', 13);
for key = fieldnames(defaults).'
    k = key{1};
    if ~isfield(s, k)
        s.(k) = defaults.(k);
    end
end
rng(s.seed);

%---------------------------------------------
% Create model for preprocessing
%---------------------------------------------
if ischar(file_or_obj)
    settings = struct(...
        'to_subtract_mean',false,...
        'to_subtract_mean_sparse',false,...
        'to_subtract_mean_global',false,...
        'add_constant_signal',false,...
        'use_deriv',false,...
        'augment_data', 7,...
        'to_add_stimulus_signal', false,...
        ...'designated_controller_channels', {{'sensory', 1}},...
        'filter_window_dat', 1,...
        'dmd_mode','no_dynamics',...
        ...'global_signal_subset', {{'DT'  'VT'  'REV'  'FWD'  'SLOW'}},...
        ...'autocorrelation_noise_threshold', 0.6,...
        'lambda_sparse',0);
    settings.global_signal_mode = 'ID_binary_transitions';

    my_model_time_delay = CElegansModel(file_or_obj, settings);
else
    assert(isa(file_or_obj, 'CElegansModel'), 'Wrong object type')
    my_model_time_delay = file_or_obj;
end

settings_ideal = my_model_time_delay.settings;

%---------------------------------------------
%% Iteratively build sparse models and remove neurons
%---------------------------------------------
max_iter = s.max_iter;
which_fit = s.which_fit; %TODO

U2 = my_model_time_delay.control_signal(:,2:end);
X1 = my_model_time_delay.dat(:,1:end-1);
n = my_model_time_delay.original_sz(1);
num_ctr = size(U2,1);
elimination_pattern = false(num_ctr, size(X1,1), max_iter);
all_thresholds_3d = zeros(num_ctr, max_iter);
disp('Calculating elimination path for each controller...')
% To export
B_prime_lasso_td_3d = zeros(size(elimination_pattern));
all_intercepts_td = zeros(num_ctr, max_iter);
elimination_neurons = zeros(num_ctr, max_iter);
all_err = zeros(num_ctr, max_iter);
all_fp = zeros(num_ctr, max_iter);
all_fn = zeros(num_ctr, max_iter);
num_spikes = zeros(num_ctr, 1);

ctr = my_model_time_delay.control_signal;

to_fit_all_models = s.to_fit_all_models;
if isnumeric(s.which_single_signal)
    which_single_model = s.which_single_signal;
elseif ischar(s.which_single_signal) || ischar(s.which_single_signal)
    key = intersect(my_model_time_delay.state_labels_key, ...
        my_model_time_delay.global_signal_subset, 'stable');
    which_single_model = find(ismember(key, s.which_single_signal));
else
    which_single_model = 1; % Still fits all models
end

for i = 1:max_iter
    fprintf('Iteration %d...\n', i)
    % Remove the top neurons from the last round
    if i > 1
        [~, top_ind] = max(abs(B_prime_lasso_td_3d(:, :, i-1)), [], 2);
        % We'll get a single time slice of a neuron, but want to remove all
        % copies (cumulatively)
        top_ind = mod(top_ind-1,n)+1 + n*(0:settings_ideal.augment_data-1);
        elimination_neurons(:,i) = top_ind(:,1);
        elimination_pattern(:,:,i) = elimination_pattern(:,:,i-1);
        for i4 = 1:size(top_ind,1)
            elimination_pattern(i4,top_ind(i4,:),i) = true;
        end
    end
    % Fit new Lasso models
    for i2 = 1:num_ctr
        if ~to_fit_all_models
            i2 = which_single_model; %#ok<FXSET>
        end
        if s.to_use_LASSO
            this_X1 = X1;
            this_X1(elimination_pattern(i2,:,i),:) = 0;
            [all_fits, fit_info] = lasso(this_X1', U2(i2,:), 'NumLambda',5);
            B_prime_lasso_td_3d(i2, :, i) = all_fits(:,which_fit); % Which fit = determined by eye
            all_intercepts_td(i2, i) = fit_info.Intercept(which_fit);
        else
            error('Only LASSO implemented currently')
        end
        if ~to_fit_all_models
            break;
        end
    end
    % Get the reconstructions of the control signals
    ctr_tmp = B_prime_lasso_td_3d(:, :, i) * X1;
    ctr_reconstruct_td = [ctr(:,1), ctr_tmp + all_intercepts_td(:, i)];
    
    if ~s.to_calculate_false_detection
        continue
    end
    for i2 = 1:num_ctr
        if ~to_fit_all_models
            i2 = which_single_model;
        end
        % Find a threshold which is best for the all-neuron
        % reconstruction
        f = @(x) minimize_false_detection(ctr(i2,:), ...
            ctr_reconstruct_td(i2,:), x, 0.1);
        all_thresholds_3d(i2, i) = fminsearch(f, 1);
        % Old style flat threshold
        [all_fp(i2,i), all_fn(i2,i), num_spikes(i), ...
            ~, ~, true_pos, ~, ~, true_neg] = ...
            calc_false_detection(ctr(i2,:), ctr_reconstruct_td(i2,:),...
            all_thresholds_3d(i2, i), [], [], false, true, false);
        % "new" findpeaks on derivative version
%         [all_fp(i2,i), all_fn(i2,i), num_spikes(i)] = ...
%             calc_false_detection(ctr(i2,:), ctr_reconstruct_td(i2,:),...
%             all_thresholds_3d(i2, i), [], [], false, true);
        % Get the error metric
        if ~isempty(s.which_error_metric)
            % Only want to calculate the first two a single time, thus the
            % additional equality
            if strcmp(s.which_error_metric, 'correlation') ...
                    && i2 == which_single_model
                this_corr = corrcoef([ctr_reconstruct_td' ctr']);
                all_err(:,i) = diag(this_corr, num_ctr);
            elseif strcmp(s.which_error_metric, 'L2') ...
                    && i2 == which_single_model
                all_err(:,i) = vecnorm(ctr_reconstruct_td - ctr, 2, 2);
            elseif strcmp(s.which_error_metric, 'mcc') 
                all_err(i2,i) = calc_mcc(...
                    true_pos, true_neg, all_fp(i2,i), all_fn(i2,i));
            end
        end
        if ~to_fit_all_models
            break;
        end
    end
end

% Get names of neurons
a = arrayfun(@(x)my_model_time_delay.get_names(x,true),...
    elimination_neurons(:,2:end), 'UniformOutput',false);
eliminated_names = cellfun(@(x)x{1},a,'UniformOutput',false);

end