classdef AdaptiveDmdc < AbstractDmd
    % Adaptive DMD with control
    %   In a heuristic way, learns which data entries are acting as controllers
    %   for the rest of the system, and separates them out as 'u_indices'
    %
    % INPUTS
    %   INPUT1 -
    %   INPUT2 -
    %
    % OUTPUTS -
    %   OUTPUT1 -
    %   OUTPUT2 -
    %
    %
    % Dependencies
    %   .m files, .mat files, and MATLAB products required:(updated on 10-Apr-2018)
    %             MATLAB (version 9.2)
    %             Signal Processing Toolbox (version 7.4)
    %             Curve Fitting Toolbox (version 3.5.5)
    %             System Identification Toolbox (version 9.6)
    %             Optimization Toolbox (version 7.6)
    %             Simulink Control Design (version 4.5)
    %             Statistics and Machine Learning Toolbox (version 11.1)
    %             Computer Vision System Toolbox (version 7.3)
    %             v2struct.m
    %             settings_importable_from_struct.m
    %             AbstractDmd.m
    %             plotSVD.m
    %             plot_2imagesc_colorbar.m
    %             xgeqp3_m.mexw64
    %             xormqr_m.mexw64
    %             optdmd.m
    %             varpro2.m
    %             varpro2dexpfun.m
    %             varpro2expfun.m
    %             varpro_lsqlinopts.m
    %             varpro_opts.m
    %             adjustedVariance.m
    %             computeTradeOffCurve.m
    %             invPow.m
    %             optimizeVariance.m
    %             sparsePCA.m
    %
    %   See also: OTHER_FUNCTION_NAME
    %
    %
    %
    % Author: Charles Fieseler
    % University of Washington, Dept. of Physics
    % Email address: charles.fieseler@gmail.com
    % Website: coming soon
    % Created: 11-Mar-2018
    %========================================
    
    properties (SetAccess={?SettingsImportableFromStruct}, Hidden=true)
        % Detection of control neurons (can be set manually)
        sort_mode
        u_sort_ind
        x_indices
        id_struct
        % Plotting options
        to_plot_nothing
        to_plot_cutoff
        to_plot_data
        to_plot_A_matrix_svd
        which_plot_data_and_filter
        to_plot_data_and_outliers
        
        dmd_mode
        sparsity_goal % if dmd_mode==sparse
        to_print_error
        % Outlier calculation settings
        error_tol
        min_number_outliers
        cutoff_multiplier
        % Preprocessing settings
        to_normalize_envelope
        filter_window_size
        outlier_window_size
        % Augmentation settings
        to_augment_error_signals
        % Tolerance for sparsification
        sparse_tol
    end
    
    properties (SetAccess=public)
        u_indices
        sep_error
        original_error
        error_mat
        error_outliers
        neuron_errors
        % DMD propagators
        A_original
        A_separate
        % Sparsified versions of A
        A_thresholded
        use_A_thresholded
        
        x_len
        u_len
    end
    
    methods
        function self = AdaptiveDmdc( file_or_dat, settings )
            % Creates adaptive_dmdc object using the filename or data
            % matrix (neurons=rows, time=columns) in file_or_dat.
            % The settings struct can have many fields, as explained in the
            % full help command.
            %
            % This initializer runs the following functions:
            %   import_settings_to_self(settings);
            %   preprocess();
            %   calc_data_outliers();
            %   calc_outlier_indices();
            %   calc_dmd_and_errors();
            %   plot_using_settings();
            % 
            % And optionally:
            %   augment_and_redo();
            
            %% Set defaults and import settings
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            self.use_A_thresholded = false;
            if self.to_plot_nothing
                self.turn_plotting_off(); %Overrides other options
            end
            %==========================================================================
            
            %---------------------------------------------
            %% Import data
            %---------------------------------------------
            if ischar(file_or_dat)
                %     self.filename = file_or_dat;
                self.raw = importdata(file_or_dat);
                %     if isstruct(tmp_struct)
                %         self.import_from_struct(tmp_struct);
                %     else
                %         error('Filename must contain a struct')
                %     end
            elseif isnumeric(file_or_dat)
                self.raw = file_or_dat;
            else
                error('Must pass data matrix or filename')
            end
            self.preprocess();
            
            %---------------------------------------------
            %% Get outlier indices
            %---------------------------------------------
            % Note: sets the variable 'x_indices' (not logical)
            % Also reorders the data so that the outliers are on the bottom
            % rows
            self.calc_data_outliers();
            self.calc_outlier_indices();
            
            %---------------------------------------------
            %% Do normal DMD and 'separated DMD'; plot
            %---------------------------------------------
            self.calc_dmd_and_errors();
            self.plot_using_settings();
            if self.sparse_tol > 0
                self.set_A_thresholded(self.sparse_tol);
            end
            if self.to_augment_error_signals
                self.augment_and_redo();
            end
            
            if self.verbose
                fprintf('Finished analyzing\n')
            end
        end
        
        function set_defaults(self)
            
            defaults = struct(...
                'sort_mode', 'DMD_error_outliers',...
                'to_plot_nothing', false,...
                'to_plot_cutoff', true,...
                'to_plot_data', true,...
                'to_plot_A_matrix_svd', false,...
                'which_plot_data_and_filter', 0,...
                'to_plot_data_and_outliers', false,...
                'dmd_mode', 'naive',...
                'sparsity_goal', 0.6,...
                'cutoff_multiplier', 1.0,...
                'to_print_error', false, ...
                'error_tol', 1e-8,...
                'min_number_outliers', 2,...
                'to_subtract_mean', false,...
                'to_normalize_envelope', false,...
                'to_augment_error_signals', false,...
                'filter_window_size', 10,...
                'outlier_window_size', 2000,...
                'id_struct', struct(),...
                'sparse_tol', 0);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
        end
        
        function calc_data_outliers(self)
            % Calculate which neurons have nonlinear features and should be
            % separated out as controllers by various methods:
            %
            %   sparsePCA: take out the nodes that have high loading on the
            %       first 15 pca modes
            %   
            % The following methods all solve the DMD problem (x2=A*x1),
            % subtract off the resulting linear fit, and then calculate
            % features based on the residuals
            %   DMD_error: choose neurons with the highest L2 error
            %   DMD_error_exp: choose neurons with the highest error
            %       calculated using an exponential (length scale,
            %       'lambda', set in the initial options)
            %   DMD_error_normalized: choose neurons with high L2 error,
            %       normalized by their median activity
            %   DMD_error_outliers: (default) choose neurons with high
            %       'outlier error,' which is the L2 norm of only outlier
            %       points (>=3 std dev away) weighted by 1/distance to
            %       neighbors (i.e. clusters are weighted more strongly)
            %
            %   random: randomly selects neurons; for benchmarking
            %   user_set: The user sets the 'x_indices' setting manually
            %       (Note: also requires setting 'sort_mode')
            %
            % Note: if the setting 'to_plot_cutoff' is true, then this
            % function plots the errors of each neuron and the cutoffs used
            % as an interactive graph
            
            X = self.dat;
            self.error_outliers = [];
            
            switch self.sort_mode
                case 'sparsePCA'
                    %---------------------------------------------
                    % Do sparse PCA to sort nodes
                    %---------------------------------------------
                    cardinality = 50;
                    num_modes = 15;
                    [modes, loadings] = sparsePCA(X',...
                        cardinality, num_modes, 2, 0);
                    loading_log_cutoff = 4;
                    loading_cutoff_index = find(...
                        log(abs(diff(loadings)))<loading_log_cutoff,1) - 1;
                    
                    % Plot the cutoff as sanity check
                    if self.to_plot_cutoff
                        figure;
                        plot(loadings)
                        hold on;
                        plot(loading_cutoff_index, loadings(loading_cutoff_index), 'r*')
                        title('sparsePCA cutoff')
                    end
                    
                    x_ind = [];
                    for j=1:loading_cutoff_index
                        x_ind = [x_ind; find(modes(:,j)>0)]; %#ok<AGROW>
                    end
                    x_ind = unique(x_ind);
                    
                case 'DMD_error'
                    X1_original = X(:,1:end-1);
                    X2_original = X(:,2:end);
                    A = X2_original / X1_original;
                    
                    self.neuron_errors = mean((A*X1_original-X2_original).^2,2);
                    cutoff_val = (mean(self.neuron_errors) + ...
                        std(self.neuron_errors)*self.cutoff_multiplier);
                    x_ind = find(self.neuron_errors < cutoff_val);
                    
                    % Plot the cutoff as sanity check
                    if self.to_plot_cutoff
                        figure;
                        plot(self.neuron_errors)
                        hold on;
                        vec = ones(size(self.neuron_errors'));
                        plot(cutoff_val*vec, 'r')
                        title('DMD reconstruction error cutoff')
                    end
                    
                case 'DMD_error_exp'
                    lambda = 0.05; % For Zimmer data
                    X1_original = X(:,1:end-1);
                    X2_original = X(:,2:end);
                    A = X2_original / X1_original;
                    
                    self.neuron_errors = sum(exp(...
                        abs(A*X1_original-X2_original)/lambda),2);
                    
                    cutoff_val = (mean(self.neuron_errors) + ...
                        std(self.neuron_errors)*self.cutoff_multiplier);
                    x_ind = find(self.neuron_errors < cutoff_val);
                    
                    % Plot the cutoff as sanity check
                    if self.to_plot_cutoff
                        figure;
                        plot(self.neuron_errors)
                        hold on;
                        vec = ones(size(self.neuron_errors'));
                        plot(cutoff_val*vec, 'r')
                        title('DMD reconstruction error cutoff')
                    end
                    
                case 'DMD_error_normalized'
                    X1_original = X(:,1:end-1);
                    X2_original = X(:,2:end);
                    A = X2_original / X1_original;
                    
                    neuron_errors_tmp = mean((A*X1_original-X2_original).^2,2);
                    self.neuron_errors = neuron_errors_tmp./(mean(X,2)+mean(mean(X)));
                    
                    cutoff_val = ( mean(self.neuron_errors) + ...
                        std(self.neuron_errors)*self.cutoff_multiplier );
                    x_ind = find(self.neuron_errors < cutoff_val);
                    % Plot the cutoff as sanity check
                    if self.to_plot_cutoff
                        figure;
                        plot(self.neuron_errors)
                        hold on;
                        vec = ones(size(self.neuron_errors'));
                        plot(cutoff_val*vec, 'r')
                        title('DMD reconstruction error cutoff')
                    end
                    
                case 'DMD_error_outliers'
                    X1_original = X(:,1:end-1);
                    X2_original = X(:,2:end);
                    A = X2_original / X1_original;
                    self.neuron_errors = A*X1_original-X2_original;
                    
                    if max(max(self.neuron_errors)) < self.error_tol
                        warning('Fitting errors lower than tolerance; no control signals can be reliably identified')
                        x_ind = true(size(self.neuron_errors,1),1);
                        self.error_outliers = zeros(size(x_ind));
                    else
                        self.error_outliers = self.calc_error_outliers(self.neuron_errors, ...
                            self.filter_window_size, self.outlier_window_size);
                        x_ind = find(~isoutlier(self.error_outliers,...
                            'ThresholdFactor', self.cutoff_multiplier));
                    end
                    
                    if self.to_plot_cutoff
                        self.plot_data_and_outliers(self.error_outliers,[],true);
                        title('Error signal detected and threshold')
                    end
                    
                case 'DMD_error_outliers_sparse'
                    % External function that uses cvx
                    error('Need to update the syntax here')
                    [ A_sparse, this_error ] = ...
                        sparse_dmd( X, min_tol, max_error_mult );
                    
                    self.neuron_errors = this_error;
                    
                    if max(max(self.neuron_errors)) < self.error_tol
                        warning('Fitting errors lower than tolerance; no control signals can be reliably identified')
                        x_ind = true(size(self.neuron_errors,1),1);
                        self.error_outliers = zeros(size(x_ind));
                    else
                        self.error_outliers = self.calc_error_outliers(self.neuron_errors, ...
                            self.filter_window_size, self.outlier_window_size);
                        x_ind = find(~isoutlier(self.error_outliers,...
                            'ThresholdFactor', self.cutoff_multiplier));
                    end
                    
                    if self.to_plot_cutoff
                        self.plot_data_and_outliers(self.error_outliers,[],true);
                        title('Error signal detected and threshold')
                    end
                    
                case 'random'
                    tmp = randperm(size(X,1));
                    x_ind = tmp(1:round(size(X,1)-...
                        self.min_number_outliers*self.cutoff_multiplier));
                    
                case 'user_set'
                    x_ind = self.x_indices;
                    
                otherwise
                    error('Sort mode not recognized')
            end
            
            if isempty(self.error_outliers)
                % Just for later plotting
                X1_original = X(:,1:end-1);
                X2_original = X(:,2:end);
                A = X2_original / X1_original;
                self.neuron_errors = A*X1_original-X2_original;
                self.error_outliers = self.calc_error_outliers(self.neuron_errors, ...
                            self.filter_window_size, self.outlier_window_size);
            end
            
            self.x_indices = x_ind;
        end
        
        function calc_outlier_indices(self)
            % Given indices that should be used for the data, calculate and
            % save the controller indices as well (the complement of the
            % data indices).
            %   Important: also sorts the data matrix so that the
            %   controllers are the last rows
            
            x_ind = self.x_indices;
            X = self.dat;
            %---------------------------------------------
            % Finish analyzing the x_indices variable
            %---------------------------------------------
            u_ind = true(size(X,1),1);
            u_ind(x_ind) = false;
            
            u_length = length(find(u_ind));
            x_length = length(find(x_ind));
            
            % Sort the data: control signals in last columns
            [~, self.u_sort_ind] = sort(u_ind);
            X = X(self.u_sort_ind,:);
            self.dat = X;
            
            self.u_len = u_length;
            self.x_len = x_length;
            self.u_indices = u_ind;
            
        end
        
        function calc_dmd_and_errors(self)
            % Actually performs dmd and calculates errors
            %
            % The basic DMD algorithm solves for _A_ in the equation: 
            %   $$ x2 = A*x1 $$
            % where the original data matrix, _X_, has been split:
            %   $$ x2 = X(:,2:end) $$
            %   $$ x1 = X(:,1:end-1) $$
            %
            % DMD with control adds an additional layer, and solves for _A_
            % and _B_ in the equation:
            %   $$ x2 = A*x1 + B*u $$
            % where _u_ is the control signal and _B_ is the connection
            % between the control signal and the dynamics in the data. In
            % this class, the control signal is taken to be certain rows
            % of the data as identified by the setting 'sort_mode'
            %
            % TODO: implement better and less biased algorithms for DMD
            
            x_length = self.x_len;
            u_length = self.u_len;
            X = self.dat;
            X1_original = X(:,1:end-1);
            X2_original = X(:,2:end);
            
            switch self.dmd_mode
                case 'naive'
                    A_orig = X2_original / X1_original;

                    % Set points corresponding to u on the LHS to 0
                    % X2_sep(u_indices,:) = 0;
                    % Easier with sorted data
                    sz = size(X2_original);
                    X2_sep = [X2_original(1:x_length,:);
                        zeros(u_length,sz(2)) ];
                    A_sep = X2_sep / X1_original;
                    
                case 'sparse'
                    p.min_tol = 8e-3;
                    p.max_error_mult = 1.75;
                    p.column_mode = true;
                    p.rows_to_predict = self.x_len;
                    p.verbose = true;
                    p.error_func = @(A, ~, ~) ...
                        self.update_and_reconstruction_error(A,'Inf');
                    p.sparsity_goal = self.sparsity_goal;
                    p.max_iter = 50;
                    
                    [ A_orig, ~ ] = ...
                        sparse_dmd( X, p);

                    % Set points corresponding to u on the LHS to 0
                    % X2_sep(u_indices,:) = 0;
                    % Easier with sorted data
                    sz = size(X2_original);
                    X2_sep = [X2_original(1:x_length,:);
                        zeros(u_length,sz(2)) ];
%                     [ A_sep, ~ ] = ...
%                         sparse_dmd( {X1_original, X2_sep},...
%                         min_tol, max_error_mult, column_mode );
                    A_sep = [A_orig(1:x_length,:);...
                        zeros(self.u_len, size(A_orig,2))];
                    
                case 'optdmd' 
                    % Note: default rank is high, so it takes a while
                    t = (1:size(X,1))';
                    r = round(size(X,1)/2);
                    %                 r = size(X,1);
                    [w,e,b,atilde,u,A_orig] = optdmd(X',t,r, 1);
                    A_sep = [A_orig(1:x_length,:);...
                        zeros(x_length, size(A_orig,2))];
                otherwise
                    error('Unrecognized dmd mode')
            end
            
            self.error_mat = A_orig*X1_original-X2_original;
            self.original_error = norm(self.error_mat)/numel(X2_original);
            % Note that the separated DMD is attempting to reconstruct a smaller set of
            % data
            self.sep_error = ...
                norm(A_sep*X1_original-X2_sep)/(x_length*size(X2_sep,2));

            self.A_original = A_orig;
            self.A_separate = A_sep;
        end
        
        function augment_and_redo(self)
            % Augments the data with the error signals from the first
            % run-through, and then redoes the analysis
            
            % Current dat has no pseudo-neurons... set it as the new data
            % to reconstruct
            self.x_indices = 1:size(self.dat,1);
            self.sort_mode = 'user_set';
            self.min_number_outliers = self.u_len;
            augmented_dat = [self.dat(:,2:end);...
                self.error_mat(self.x_len+1:end,:)];
            
            % Set the new data, and redo all initializer steps
            self.raw = augmented_dat;
            self.preprocess();
            self.calc_data_outliers();
            self.calc_outlier_indices();
            self.calc_dmd_and_errors();
            self.plot_using_settings();
        end
        
        function error_outliers = calc_error_outliers(self, neuron_errors,...
                filter_window_size, outlier_window_size)
            
            error_outliers = zeros(size(neuron_errors,1),1);
            tspan = 1:size(neuron_errors,2);
            
            for i = 1:size(neuron_errors,1)
                this_error = neuron_errors(i,:)';
                this_error = filter(ones(filter_window_size,1)/filter_window_size,...
                    1, this_error);
                %             this_error = this_error/var(this_error);
                if self.to_normalize_envelope
                    [up_env,low_env] = envelope(this_error);
                    this_error = this_error /...
                        mean(abs([mean(up_env) mean(low_env)]));
                end
                outlier_indices = isoutlier(this_error,...
                    'movmedian',outlier_window_size,'SamplePoints',tspan);
                % 2 factors increasing the importance of outliers:
                %   magnitude (i.e. L2 error... first term)
                %   clusters (i.e. shorter distance between neighbors... second term)
                this_pts = this_error(outlier_indices);
                if length(find(outlier_indices))<self.min_number_outliers
                    error_outliers(i) = 0;
                    continue
                end
                if length(find(outlier_indices))>1
                    n_dist = diff(find(outlier_indices));
                    neighbor_weights = [n_dist; n_dist(end)] + [n_dist(1); n_dist];
                    
                    error_outliers(i) = sum( (this_pts .* (1./neighbor_weights)).^2 );
                else
                    error_outliers(i) = sum( 2*(this_pts.^2) / length(tspan) );
                end
            end
        end
        
        function pruned_ad_obj = prune_outliers(self, error_tol)
            % Prunes control signals from the list greedily as long as the
            % reconstruction error is below tolerance
            initial_error = self.calc_reconstruction_error();
            if ~exist('error_tol','var')
                error_tol = 2*initial_error;
            elseif error_tol<initial_error
                warning('Error tolerance below current error; no pruning possible')
                pruned_ad_obj = [];
                return
            end
            
            fprintf('Initial error is %f\n', initial_error);
            test_settings = self.settings;
            test_settings.sort_mode = 'user_set';
            test_settings.to_plot_nothing = true;
            
            new_errors = zeros(self.u_len,1);
            for i=1:self.u_len
                % Remove current control nodes 1 by 1; get new error
                %   Note: self.dat is sorted with control signals at
                %   the bottom
                test_settings.x_indices = [1:self.x_len self.x_len+i];
                this_ad_obj = adaptive_dmdc(self.dat, test_settings);
                new_errors(i) = this_ad_obj.calc_reconstruction_error();
                fprintf('New error is %f\n', new_errors(i));
            end
            [~, sorted_error_ind] = sort(new_errors);
            
            % Definitely remove the least error-contributing signal
            test_x_indices = [1:self.x_len ...
                self.x_len+sorted_error_ind(1)];
            for i = 2:self.u_len
                this_indices = [test_x_indices ...
                    self.x_len+sorted_error_ind(i)];
                test_settings.x_indices = this_indices;
                this_ad_obj = adaptive_dmdc(self.dat, test_settings);
                
                this_error = this_ad_obj.calc_reconstruction_error();
                if this_error > error_tol
                    fprintf('Final error: %f\n',...
                        pruned_ad_obj.calc_reconstruction_error())
                    fprintf('Final control set size: %d\n',...
                        pruned_ad_obj.u_len);
                    break
                else
                    test_x_indices = this_indices;
                    pruned_ad_obj = this_ad_obj;
                end
            end
            
            
        end
        
        function names = get_names(self, neuron_ind, ...
                use_original_order, print_names, print_warning, ...
                to_parse_names)
            % Gets and optionally prints the names of the passed neuron(s).
            % Calls recursively if a list is passed
            %   Note: many will not be identified uniquely; the default is
            %   to just concatenate the names
            if isempty(fieldnames(self.id_struct))
                disp('Names not stored in this object; aborting')
                names = 'Neuron names not stored';
                return
            elseif isempty(self.id_struct.ID)
                disp('Names not stored in this object; aborting')
                names = 'Neuron names not stored';
                return
            end
            if ~exist('use_original_order','var')
                use_original_order = true;
            end
            if ~exist('print_names','var')
                print_names = true;
            end
            if ~exist('print_warning','var')
                print_warning = true;
            end
            if ~exist('to_parse_names','var')
                to_parse_names = true;
            end
            
            % Call recursively if a list is input
            if ~isscalar(neuron_ind)
                names = cell(size(neuron_ind));
                for n=1:length(neuron_ind)
                    this_neuron = neuron_ind(n);
                    names{n} = ...
                        self.get_names(this_neuron, ...
                        use_original_order, print_names, print_warning);
                end
                return
            end
            
            % The data might be sorted, so get the new index
            if ~use_original_order
                neuron_ind = self.u_sort_ind(neuron_ind);
            end
            
            % Actually get the name
            if neuron_ind > length(self.id_struct.ID)
                if print_warning
                    warning('Index outside the length of names; assuming derivatives')
                end
                neuron_ind = neuron_ind - length(self.id_struct.ID);
                if neuron_ind > length(self.id_struct.ID)
                    names = '';
                    return
                end
            end
            names = {self.id_struct.ID{neuron_ind},...
                self.id_struct.ID2{neuron_ind},...
                self.id_struct.ID3{neuron_ind}};
            if print_names
                fprintf('Identifications of neuron %d: %s, %s, %s\n',...
                    neuron_ind, names{1},names{2},names{3});
            end
            if to_parse_names
                names = self.parse_names({names});
                if length(names)==1
                    % i.e. a single neuron
                    names = names{1};
                end
            end
            
        end % function
    end % methods
    
    methods % Reconstruction
        function dat_approx = calc_reconstruction_original(self, x0, tspan)
            % Reconstructs the data using the full DMD propagator matrix
            % starting from the first data point
            if ~exist('x0','var') || isempty(x0)
                x0 = self.dat(:,1);
            end
            if ~exist('tspan','var')
                num_frames = size(self.dat,2);
                tspan = linspace(0,num_frames*self.dt,num_frames+1);
            end
            
            A = self.A_original;
            
            dat_approx = zeros(length(x0), length(tspan));
            dat_approx(:,1) = x0;
            for i=2:length(tspan)
                dat_approx(:,i) = A * dat_approx(:,i-1);
            end
        end
        
        function dat_approx = calc_reconstruction_control(self,...
                x0, tspan, include_control_signal)
            % Reconstructs the data using the partial DMD propagator matrix
            % starting from the first data point using the 'outlier' rows
            % as control signals
            if ~exist('x0','var') || isempty(x0)
                x0 = self.dat(:,1);
            end
            if ~exist('tspan','var') || isempty(tspan)
                num_frames = size(self.dat,2);
                tspan = linspace(0,num_frames*self.dt,num_frames);
            end
            if ~exist('include_control_signal','var')
                include_control_signal = false;
            end
            
            ind = 1:self.x_len;
            if ~self.use_A_thresholded
                A = self.A_separate(ind, ind);
                B = self.A_separate(ind, self.x_len+1:end);
            else
                A = self.A_thresholded(ind, ind);
                B = self.A_thresholded(ind, self.x_len+1:end);
            end
            
            if include_control_signal
                dat_approx = zeros(length(x0), length(tspan));
                % bottom rows are not reconstructed; taken as given
                dat_approx(self.x_len+1:end,:) = ...
                    self.dat(self.x_len+1:end,:);
                dat_approx(:,1) = x0;
            else
                dat_approx = zeros(self.x_len, length(tspan));
                dat_approx(:,1) = x0(1:self.x_len);
            end
            for i=2:length(tspan)
                u = self.dat(self.x_len+1:end, i);
                dat_approx(1:self.x_len, i) = ...
                    A*dat_approx(1:self.x_len, i-1) + B*u;
            end
        end
        
        function error_approx = calc_reconstruction_error(self, ...
                which_norm, varargin)
            % Does not include the control signal
            % User can specify a particular mode (2-norm, etc)
            if ~exist('which_norm','var') || isempty(which_norm)
                which_norm = '2norm';
            end
            % Does not include the control signal (which has 0 error)
            dat_approx = self.calc_reconstruction_control([],[],false);
            dat_original = self.dat(1:self.x_len,:);
            
            switch which_norm
                case '2norm'
                    error_approx = ...
                        norm(dat_approx-dat_original)/numel(dat_approx);
                case 'Inf'
                    error_approx = ...
                        norm(dat_approx-dat_original,Inf)/numel(dat_approx);
                case 'expnorm'
                    lambda = varargin{1};
                    assert(lambda>0, 'The length scale should be positive')
                    error_approx = sum(sum(...
                        exp((dat_approx-dat_original)./lambda) )) / ...
                        numel(dat_approx);
                case 'flat_then_2norm'
                    error_approx = dat_approx-dat_original;
                    if nargin < 3
                        lambda = mean(var(error_approx));
                    else
                        lambda = varargin{1};
                    end
                    set_to_zero_ind = abs(error_approx)<lambda;
                    error_approx(set_to_zero_ind) = 0;
                    error_approx = norm(error_approx) / ...
                        length(find(set_to_zero_ind));
                    
                otherwise
                    error('Unknown error metric')
            end
        end
        
        function A = set_A_thresholded(self, tol)
            % Thresholded A_original and changes the reconstruction
            % plotters to use that matrix
            self.use_A_thresholded = true;
            self.sparse_tol = tol;
            
            A = self.A_original;
            A(abs(A)<tol) = 0;
            self.A_thresholded = A;
        end
        
        function reset_threshold(self)
            % Resets settings back to original dynamics matrix
            self.use_A_thresholded = false;
        end
    end
    
    methods (Access=private)
        function error_approx = update_and_reconstruction_error(self,...
                A, which_norm, varargin)
            % Updates the A_sep matrix (meant to be temporary, e.g. in the
            % midst of converging on a final A_sep), then calculates the
            % error and returns it (scalar)
            if ~exist('which_norm','var') || isempty(which_norm)
                which_norm = '2norm';
            end
            if ~exist('varargin','var')
                varargin = {};
            end
            self.A_separate = A;
            error_approx = ...
                self.calc_reconstruction_error(which_norm, varargin);
        end
    end
    
    methods % Plotting
        
        function plot_using_settings(self)
            % Plots several things after analysis is complete
            %
            % There are several plotting possibilities, which are all set
            %       in the original settings struct:
            % 'to_plot_data': the sorted data set (control signals on 
            %   bottom)
            % 'to_plot_A_matrix_svd': the svd of the matrix which solves 
            %   the equation $$ x_(t+1) = A*x_t $$ will be plotted
            % 'which_plot_data_and_filter': plots certain neurons with the
            %   filter used to determine error outliers
            % 'to_plot_data_and_outliers': plots individual neuron errors
            %   with outliers marked (method depends on 'sort_mode');
            %   interactive
            %
            % And printing options:
            % 'to_print_error': prints L2 error of normal fit vs. fit using
            %   the control signals
            % 'use_optdmd': also prints L2 error of an alternative dmd
            %   algorithm
            
            x_length = self.x_len;
            u_length = self.u_len;
            
            %---------------------------------------------
            %% Plotting options
            %---------------------------------------------
            if self.to_plot_data
                self.plot_data_and_control();
            end
            
            if self.to_plot_A_matrix_svd
                A_x = self.A_separate(1:x_length,1:x_length);
                A_u = self.A_separate(1:x_length,x_length+1:end);
                
                plotSVD(A_x);
                title(sprintf('Max possible rank (data): %d',x_length))
                
                plotSVD(A_u);
                title(sprintf('Max possible rank (ctr): %d',min(u_length,x_length)))
            end
            
            %---------------------------------------------
            %% Different per-neuron error metrics
            %---------------------------------------------
            
            if self.which_plot_data_and_filter > 0
                self.plot_data_and_filter(...
                    self.error_mat(self.which_plot_data_and_filter,:)',...
                    self.filter_window_size, self.outlier_window_size)
            end
            if self.to_plot_data_and_outliers
                sorted_error_outliers = self.calc_error_outliers(self.error_mat, ...
                    self.filter_window_size, self.outlier_window_size);
                self.plot_data_and_outliers(sorted_error_outliers,...
                    @(x,y) self.callback_plotter(x,y, ...
                    self.error_mat, self.filter_window_size, self.outlier_window_size),...
                    false)
            end
            
            if self.to_print_error
                fprintf('Error in original dmd is %f\n',self.original_error)
                fprintf('Error in separated dmd is %f\n',self.sep_error)
            end
            
            if strcmp(self.dmd_mode, 'optdmd')
                optdmd_error = norm(w*diag(b)*exp(e*t)-X)/numel(X);
                fprintf('Error in optdmd is %f\n',optdmd_error)
            end
            %==============================================================
        end
        
        function plot_data_and_control(self)
            % Plots the raw data on the left and the data with the control
            % signal set to 0 on the right
                X2_sep = ...
                    [self.dat(1:self.x_len,:);...
                    zeros(self.u_len,size(self.dat,2))];
                plot_2imagesc_colorbar(self.dat, X2_sep, '1 2',...
                    'Original X2 data (sorted)',...
                    'X2 data with u set to 0');
        end
        
        function dat_approx = plot_reconstruction(self, ...
                use_control, include_control_signal, to_compare_raw,...
                neuron_ind)
            % Plots a reconstruction of the data using the stored linear
            % model. Options (defaults in parentheses):
            %   use_control (false): reconstruct using control, i.e.
            %       $$ x_(t+1) = A*x_t + B*u $$
            %       or without control (the last _B_*_u_ term)
            %   include_control_signal (false): plot the control signal as 
            %       well as the reconstruction
            %   to_compare_raw (true): also plot the raw data
            %   neuron_ind (0): if >0, plots a single neuron instead of the
            %       entire dataset

            if ~exist('use_control','var')
                use_control = false;
            end
            if ~exist('include_control_signal','var')
                include_control_signal = false;
            end
            if ~exist('to_compare_raw', 'var')
                to_compare_raw = true;
            end
            if ~exist('neuron_ind','var')
                neuron_ind = 0;
            end
            
            if use_control
                if ~include_control_signal
                    full_dat = self.dat(1:self.x_len,:);
                    title_str = 'Reconstructed data (with control; signal not shown)';
                else
                    title_str = sprintf(...
                        'Reconstructed data (control signal = rows %d-%d)',...
                        self.x_len+1, size(self.dat,1));
                    full_dat = self.dat;
                end
                dat_approx = self.calc_reconstruction_control(...
                    [],[],include_control_signal);
            else
                full_dat = self.dat;
                dat_approx = self.calc_reconstruction_original();
                title_str = 'Reconstructed data (no control)';
            end
            if to_compare_raw
                if neuron_ind < 1
                    plot_2imagesc_colorbar(full_dat, dat_approx, '2 1',...
                        'Original data', title_str);
                else
                    title_str = [title_str ...
                        sprintf('; neuron %d (name=%s)',...
                        neuron_ind, self.get_names(neuron_ind))];
                    figure;
                    hold on
                    plot(full_dat(neuron_ind,:))
                    plot(dat_approx(neuron_ind,:), 'LineWidth',2)
                    legend({'Original data','Reconstructed trajectory'})
                    ylabel('Amplitude')
                    xlabel('Time')
                    if neuron_ind < self.x_len+1
                        title(title_str)
                    else
                        title('This neuron taken as is; no reconstruction')
                    end
                end
            else
                figure;
                if neuron_ind < 1
                    imagesc(dat_approx);
                    colorbar
                else
                    title_str = [title_str ...
                        sprintf('; neuron %d (name=%s)',...
                        neuron_ind, self.get_names(neuron_ind))];
                    plot(dat_approx(neuron_ind,:))
                    ylabel('Amplitude')
                    xlabel('Time')
                end
                title(title_str)
            end
        end
        
        function plot_data_and_filter(~, dat, filter_window, outlier_window)
            figure
            x = 1:size(dat,1);
            x_delay = x - (filter_window-1)/(2);
            plot(x, dat);
            hold on;
            dat_filter = filter(ones(1,filter_window)/filter_window,1,dat);
            TF = isoutlier(dat_filter,'movmedian',outlier_window,'SamplePoints',x);
            plot(x_delay,dat_filter, x_delay(TF),dat_filter(TF),'x','LineWidth',3)
            legend('Raw Data','Weighted Moving Average','Outlier');
        end
        
        function plot_data_and_outliers(self, ...
                dat, callback_func, use_original_order)
            % Plots the error signals with their outliers, using a GUI to
            % explore the individual neurons
            if ~exist('dat','var') || isempty(dat)
                dat = self.error_outliers;
            end
            if ~exist('callback_func','var') || isempty(callback_func)
                callback_func = @(x,y) self.callback_plotter(x,y, ...
                    self.neuron_errors,...
                    self.filter_window_size,...
                    self.outlier_window_size);
            end
            if ~exist('use_original_order','var')
                use_original_order = true;
            end
            
            figure
            x = 1:size(dat,1);
            vec = ones(1,length(x));
            hold on;
            [TF,lower,upper,center] = isoutlier(dat,...
                'ThresholdFactor', self.cutoff_multiplier);
            plot(x(TF),dat(TF),'x',...
                x,lower*vec,x,upper*vec,x,center*vec)
            plot(dat, 'o',...
                'ButtonDownFcn',callback_func);
            legend('Outlier',...
                'Lower Threshold','Upper Threshold','Center Value',...
                'Original Data')
            ylabel(sprintf('Error (measured using %s)', self.sort_mode))
            % Use neuron names for xticklabels (many will be empty)
            xticks(1:self.x_len)
            xticklabels( self.get_names(1:self.x_len,...
                use_original_order, false));
            xtickangle(90)
            if strcmp(self.sort_mode, 'user_set')
                title('Calculated outliers (user set ones used instead)')
            else
                title('Calculated outliers')
            end
        end
        
        function plot_data_and_exp_filter(~, dat, alpha, outlier_window)
            figure
            x = 1:size(dat,1);
            x_delay = x-1;
            plot(x, dat);
            hold on;
            dat_filter = filter(alpha, 1-alpha, dat);
            TF = isoutlier(dat_filter,'movmedian',outlier_window,'SamplePoints',x);
            plot(x_delay,dat_filter, x_delay(TF),dat_filter(TF),'x','LineWidth',3)
            legend('Raw Data','Weighted Moving Average','Outlier');
        end
        
        function callback_plotter(self, ~, evt, dat, filter_window, outlier_window)
            % On left click:
            %   Plots the original data minus the linear component (i.e.
            %   the error signal, interpreted as important for control)
            % On other (e.g. right) click:
            %   Displays the neuron name, if identified
            this_neuron = evt.IntersectionPoint(1);
            if evt.Button==1
                self.plot_data_and_filter(dat(this_neuron,:)',...
                    filter_window, outlier_window);
                this_name = self.get_names(this_neuron);
                if isempty(this_name)
                    this_name = 'N/A';
                end
                title(sprintf('Residual for neuron %d (name=%s)',...
                    this_neuron, this_name))
                xlabel('Time')
                ylabel('Error')
            else
                self.get_names(this_neuron);
            end
        end
        
        function turn_plotting_off(self)
            % Sets all plotting settings to 'false'
            self.to_plot_cutoff = false;
            self.to_plot_data = false;
            self.to_plot_A_matrix_svd = false;
            self.to_plot_data_and_outliers = false;
            
            self.to_print_error = false;
        end
    end
    
    methods(Static)
        function parsed_name_list = parse_names(name_list, keep_ambiguous)
            % Input: a cell array of 3x1 cell arrays, containing 3 possible
            % names for each neuron
            % Output: a cell array containing the strings of either a) only
            % unambiguously identified neurons or b) compound names when
            % the id has multiple names
            if ~exist('keep_ambiguous','var')
                keep_ambiguous = true;
            end
            
            parsed_name_list = cell(size(name_list));
            for jN=1:length(name_list)
                this_neuron = name_list{jN};
                new_name = '';
                for jID=1:length(this_neuron)
                    check_name = this_neuron{jID};
                    if isempty(check_name)
                        continue
                    elseif strcmp(check_name,new_name)
                        continue
                    else
                        % Different, non-empty ID name
                        if keep_ambiguous
                            new_name = [new_name check_name]; %#ok<AGROW>
                        else
                            new_name = 'Ambiguous';
                            break
                        end
                    end % if
                end % for
                parsed_name_list{jN} = new_name;
            end % for
        end % function
    
        function A = threshold_matrix(A, tol)
            % Sets all values in the matrix A with abs()<tol to 0
            A(abs(A)<tol) = 0;
        end
    end % methods
end % class
