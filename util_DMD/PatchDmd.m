classdef PatchDmd < handle & AbstractDmd
    %% PatchDmd
    % Performs Dynamic Mode Decomposition for patches of data
    %
    %
    % INPUTS
    %   INPUT1 -
    %   INPUT2 -
    %
    % OUTPUTS -
    %   OUTPUT1 -
    %   OUTPUT2 -
    %
    % EXAMPLES
    %
    %   EXAMPLE1
    %
    %
    %   EXAMPLE2
    %
    %
    %
    % Dependencies
    %   .m files, .mat files, and MATLAB products required:
    %
    %   See also: OTHER_FUNCTION_NAME
    %
    %
    %
    % Author: Charles Fieseler
    % University of Washington, Dept. of Physics
    % Email address: charles.fieseler@gmail.com
    % Website: coming soon
    % Created: 06-Mar-2018
    %========================================
    
    
    properties (SetAccess={?SettingsImportableFromStruct})
        % Settings for each DMDplotter object (at least one per patch)
        plotter_settings = struct('use_optdmd', true, ...
                                  'model_order', 10)
        patch_settings = struct('window_size', 20, ...
                                'window_step', 1)
        % Uses either the WindowDmd or DMDplotter class
        use_WindowDmd  = true
        use_derivatives = false
        
        % Settings for filtering patches
        min_patch_size = 20
        use_2_patches = false
    end
    
    properties (SetAccess=private) 
        % patch variables
        patch_labels
        patch_vector
        patch_starts
        patch_DMD_obj
        patch_DMD_labels
        
        patch_control_neurons
        patch_control_idx

        actually_analyzed_labels %List of which labels had enough data
        
        similarity_objects
        AdaptiveDmdc_objects
        
        dynamic_matrices_dict
        dynamic_matrices_similarities
        dynamic_matrices_average_dict
        same_label_struct
        
        cell_id_struct
    end
    
    properties (SetAccess=private, Hidden=true)
        %DMD outputs, in containers.map objects
        omega_all
        phi_all
        coeff_all
        dat_layers
        
        %Will often use this with a struct
        raw_struct
        
        %svd variables
        u
        s
        v
        proj3d
        
        % simulation variables
        tspan_full
    end
    
    methods
        
        function self = PatchDmd(file_or_dat, settings)
            %% Initialize and check
            self.patch_settings.plotterSet = self.plotter_settings;
            self.import_settings_to_self(settings);
            
            self.similarity_objects = containers.Map();

            self.dynamic_matrices_dict = containers.Map();
            self.dynamic_matrices_similarities = containers.Map();
            self.dynamic_matrices_average_dict = containers.Map();
            self.same_label_struct = struct();
            %==========================================================================
            
            %% Import data and preprocess
            if ischar(file_or_dat)
                self.filename = file_or_dat;
                tmp_struct = importdata(file_or_dat);
                if isstruct(tmp_struct)
                    self.import_from_struct(tmp_struct);
                else
                    error('Filename must contain a struct')
                end
            elseif isstruct(file_or_dat)
                self.import_from_struct(file_or_dat);
            else
                error('Must pass struct or filename')
            end

            self.preprocess();
            %==========================================================================
            
            %% Do dmd for all patches
            self.DMD_all_patches();
            %==============================================================
            
            %% Postprocessing
            self.postprocess();
            %==============================================================

        end
        
        function import_from_struct(self, dat)
            %'dat' should be a struct
            %   hard-coded fieldnames to look for
            
            % Get the field names for the state vector and their names
            %   Also for the cell ID's; there may be multiple fields
            f_names = fieldnames(dat);
            state_strs = f_names(contains(f_names,'States'));
            % Note: above gets vector of states AND key to translate into
            % strings (IN THAT ORDER)
            ID_strs = f_names(contains(f_names,'ID'));
            id_struct = struct();
            
            %Import
            % Calcium imaging magnitudes or derivatives
            if self.use_derivatives
                self.raw = dat.tracesDif';
            else
                self.raw = dat.traces';
            end
            self.tspan_full = dat.timeVectorSeconds;
            self.dt = self.tspan_full(2) - self.tspan_full(1);
            % Patch information (hand-labeled)
            self.patch_vector = dat.(state_strs{1});
            self.patch_labels = dat.(state_strs{2});
            self.patch_starts = find(diff(self.patch_vector))+1;
            % Cell information
            for id = ID_strs'
                id_struct.(id{1}) = dat.(id{1});
            end
            self.cell_id_struct = id_struct;
            
            % Cell array for the actual analysis objects
            self.patch_DMD_obj = cell(length(self.patch_starts), 1);
            self.AdaptiveDmdc_objects = ...
                cell(length(self.patch_starts), 1);
            self.patch_DMD_labels = cell(length(self.patch_starts), 1);
        end
        
        function DMD_all_patches(self)
            %Creates a mrDMD or WindowDmd object for each patch
            
            sz = length(self.patch_starts);
            for i = 1:sz
                t_start = self.patch_starts(i);
                this_label = self.patch_labels{self.patch_vector(t_start)};
                
                % If we run off the edge (could be 1 or 2 patches ahead)
                try
                    if self.use_2_patches
                        t_end = self.patch_starts(i+2)-1;
                        this_label = [this_label '_'...
                            self.patch_labels{self.patch_vector(t_end)}]; %#ok<AGROW>
                    else
                        t_end = self.patch_starts(i+1)-1;
                    end
                catch
                    t_end = sz;
                end
                
                dat_indices = t_start:t_end;
                
                self.DMD_one_patch(i, dat_indices, this_label);
            end
            
            tmp = unique(self.patch_DMD_labels);
            self.actually_analyzed_labels = tmp(2:end); %First is ''
        end
        
        function calc_AdaptiveDmdc_all(self, dmdc_settings, to_pause)
            % Uses external AdaptiveDmdc to 'learn' the neurons that have
            % nonlinear (assumed to be control) signals
            if ~exist('dmdc_settings','var')
                dmdc_settings = struct();
            end
            if ~exist('to_pause','var')
                to_pause = false;
            end
            
            self.patch_control_neurons = [];
            for i = 1:length(self.patch_DMD_labels)
                if ~isempty(self.patch_DMD_labels{i})
                    self.calc_AdaptiveDmdc(i, dmdc_settings)
                    if to_pause
                        pause;
                    end
                end
            end
        end
        
        function [idx,C,sumd,D] = cluster_control_neurons(self, ...
                num_clusters, to_plot)
            % Clusters the binary vectors of control neurons using kmeans
            % and hamming distance
            if ~exist('to_plot','var')
                to_plot = false;
            end
            if ~exist('num_clusters','var')
                num_clusters = length(self.actually_analyzed_labels);
            end
            
            % Plot the sorted neuron labels and return the sorting matrix
            [patch_names, I] = self.sort_patch_names();
            dat = self.patch_control_neurons(:,I)';
            
            [idx, C, sumd, D] = kmeans(dat, num_clusters, 'Replicates', 5);
            % Hamming distance for binary 
%             [idx, C, sumd, D] = kmeans(dat, num_clusters, ...
%                 'Distance', 'hamming', 'Replicates', 5);
            self.patch_control_idx = idx;
            if to_plot
                self.plot_clustered_control_neurons(patch_names);
            end
            
        end
        
        function initialize_all_similarity_objects(self)
            % Initializes external object for calculating similarities
            % between DMD data reconstructions
            t_compare = self.min_patch_size/2.0;
            
            for label_cell = self.actually_analyzed_labels'
                label = label_cell{1};
                
                compare_func = self.calc_DMD_func_array(...
                    label, 'all', t_compare);
                compare_x0_mean = self.same_label_struct.starts(label);
                settings = struct('iterations',10,...
                    'noise',mean(compare_x0_mean)/10.0);
                
                self.similarity_objects(label) = ...
                    DynamicSimilarity(...
                    compare_func, compare_x0_mean, settings);
            end
        end
        
        function calc_all_label_similarities(self, to_plot)
            % Calculates the similarities within a single label
            %   Can be plotted parametrized by start time within a patch
            disp('Calculating all label similarities; may a minute or more')
            if ~exist('to_plot','var')
                to_plot = false;
            end
            if isempty(self.similarity_objects)
                self.initialize_all_similarity_objects();
            end
            for label = self.actually_analyzed_labels'
                self.calc_same_label_similarity(label{1}, to_plot)
            end
        end
        
        function calc_all_label_similarities_old(self, to_plot)
            if ~exist('to_plot','var')
                to_plot = false;
            end
            for label = self.actually_analyzed_labels'
                self.calc_same_label_similarity(label{1},'all', to_plot)
            end
        end
        
        function calc_same_label_similarity(self, label, to_plot)
            % Get similarity statistics for all windows in all patches with
            % a certain label
            % Uses external dynamic_similarity_obj object
            if ~exist('to_plot','var')
                to_plot = false;
            end
            
            % Get pairwise distance between average matrix and all others
            this_similarity_obj = self.similarity_objects(label);
            self.dynamic_matrices_similarities(label) = ...
                this_similarity_obj.calc_all_dynamic_similarity_functions();
            
            if to_plot
                self.plot_same_label_similarity(label);
            end
        end
        
        function calc_same_label_similarity_old(self,...
                label, which_patch, to_plot)
            % Get similarity statistics for all windows in all patches with
            % a certain label
            if ~exist('which_patch','var')
                which_patch = 'all';
            end
            if ~exist('to_plot','var')
                to_plot = false;
            end
            
            % The time and x0 to simulate out and compare
            t_compare = self.min_patch_size/2.0;
            
            % First, extract all matrices of the relevant patches
            % Vector indices of the patches with the right label (category)
            all_dynamic_vectors_func = self.calc_DMD_func_array(...
                label, which_patch, t_compare);
            
            % Second, get an "average" matrix
            %NOTE: all matrices really are in the same basis (neurons)
            
            % Get random cell array with n cells. Here n = 3
            num_matrices = size(all_dynamic_vectors_func,2);
            
            % Get pairwise distance between average matrix and all others
            all_similarities = zeros(num_matrices,1);
            
            for j=1:num_matrices
                all_similarities(j) = self.dynamic_similarity_functions(...
                        all_dynamic_vectors_func,...
                        all_dynamic_vectors_func{j}, ...
                        10, label, 10);
            end
            
            self.dynamic_matrices_similarities(label) = all_similarities;
            
            if to_plot
                self.plot_same_label_similarity(label);
            end
        end
        
        function calc_different_label_similarity(self,...
                label_mean, label_test, to_plot)
            % Calculates similarity between each entry of label_test and
            % the mean of label_mean
            if ~exist('to_plot','var')
                to_plot = false;
            end
            mean_funcs = self.dynamic_matrices_dict(label_mean);
            test_funcs = self.dynamic_matrices_dict(label_test);
            compound_label = [label_mean '_' label_test];
            
            num_test_matrices = size(test_funcs,2);
            % Get pairwise distance between average matrix and all others
            all_similarities = zeros(num_test_matrices,1);
            
            for j=1:num_test_matrices
                all_similarities(j) = self.dynamic_similarity_functions(...
                        mean_funcs,...
                        test_funcs{j}, ...
                        10, label_test, 10);
            end
            self.dynamic_matrices_similarities(compound_label) =...
                all_similarities;
            
            if to_plot
                self.plot_same_label_similarity(compound_label, 'tstart_box');
            end
        end
        
        function similarity = calc_different_label_similarity_mat(...
                                    self, label1, label2)
            A1 = self.dynamic_matrices_average_dict(label1);
            A2 = self.dynamic_matrices_average_dict(label2);
            similarity = self.pairwise_similarity(A1, A2);
        end
        
        function postprocess(self)
            % Calculate various class properties
            
            self.same_label_struct.indices = containers.Map();
            self.same_label_struct.within_patch_tspans = containers.Map();
            self.same_label_struct.centroids = containers.Map();
            self.same_label_struct.starts = containers.Map();
            self.same_label_struct.ends = containers.Map();
            % Dictionary of indices for all the labels that were actually
            % analyzed
            %    Also info about the centroids and average starting and
            %    ending points for each (analyzed) label
            for label = self.actually_analyzed_labels'
                label = label{1}; %#ok<FXSET>
                this_indices = ...
                    cellfun(@(x)strcmp(x,label),...
                    self.patch_DMD_labels);
                this_indices_start = (diff(this_indices)==1)+1;
                this_indices_end = (diff(this_indices)==-1);
        
                this_mean = mean(self.dat(:, this_indices), 2);
                this_starts = mean(self.dat(:, this_indices_start), 2);
                this_ends = mean(self.dat(:, this_indices_end), 2);
                
                self.same_label_struct.indices(label) = find(this_indices);
                self.same_label_struct.centroids(label) = this_mean;
                self.same_label_struct.starts(label) = this_starts;
                self.same_label_struct.ends(label) = this_ends;
            end
        end
        
        function preprocess(self)
            % Set initial data properties (mostly a stub)
            self.original_sz = size(self.raw);
            self.dat = self.raw;
            self.sz = size(self.dat);
            
            self.min_patch_size = max(self.min_patch_size,...
                self.patch_settings.windowSize);
        end
        
    end
    
    methods % Plotting functions
        
        function plot_same_label_data(self,...
                label, field, preprocess_func)
            %Plots all patches of the same label for the user-defined field
            %   The user can pass a function to preprocess the data
            %   (default is no processing)
            if ~exist('preprocess_func','var')
                preprocess_func = @(x) x;
            end
            
            %Vector indices of the patches with the right label (category)
            this_label_indices = self.same_label_struct.indices(label);
            
            for patch = this_label_indices'
                this_dat = self.patch_DMD_obj{patch}.(field);
                
                figure
                plot(preprocess_func(this_dat), 'o', 'LineWidth', 2);
                title(sprintf('Label: %s; index: %d', label, patch))
            end
        end
        
        function plot_same_label_similarity(self, label, plot_mode)
            % Plots similarity within a given label
            if ~exist('plot_mode','var')
                plot_mode = 'box';
            end
            if strfind(label,'_')>0
                t_label = label(strfind(label,'_')+1:end);
            else
                t_label=label;
            end
            figure
            switch plot_mode
                case 'box'
%                     this_obj = self.similarity_objects(label);
%                     this_obj.plot_box();
                    boxplot(self.dynamic_matrices_similarities(label))
                case 'tstart'
                    within_patch_tspans = ...
                        self.same_label_struct.within_patch_tspans(t_label);
                    plot(within_patch_tspans(1,:), ...
                        self.dynamic_matrices_similarities(label), 'o')
                    xlabel('Start time within a patch of behavior')
                case 'tstart_box'
                    within_patch_tspans = ...
                        self.same_label_struct.within_patch_tspans(t_label);
                    boxplot(self.dynamic_matrices_similarities(label), ...
                        within_patch_tspans(1,:))
                    xlabel('Start time within a patch of behavior')
                otherwise
                    error("Unrecognized plotting option")
            end
            ylabel('Dynamic similarity (max=1)')
            title(sprintf('Similarities among %s dynamics', label))
        end
        
        function plot_labelled_trace(self)
            % Plot the time series labels
            plot(self.tspan_full, self.patch_vector)
            yticklabels(self.patch_labels)
            title(self.filename)
        end
        
        function plot_SVD(self)
            % Plots svd and several related graphs
            [self.u,self.s,self.v,self.proj3d] = plotSVD(self.raw);
        end
        
        function plot_colored_trace(self)
            % Plots a colored trace of the 3d pca trajectory
            
            figure;
            hold on;
            % Get default colors + black for 'nostate'
            colors = get(gca, 'ColorOrder');
            colors = [colors; [0 0 0]];
            dat3d = self.proj3d;
            sz = length(self.patch_starts);
            for i = 1:sz
                if i<sz
                    t_end = (self.patch_starts(i+1));
                else
                    t_end = sz;
                end
                tspan = self.patch_starts(i):t_end;
                p = plot3(dat3d(1,tspan), ...
                    dat3d(2,tspan), ...
                    dat3d(3,tspan), ...
                    'LineWidth', 2);
                
                this_color_ind = self.patch_vector(...
                    self.patch_starts(i));
                set(p, 'Color', colors(this_color_ind,:))
            end
            
            title('Dynamics in the space of the first three modes')
            xlabel('mode 1'); ylabel('mode 2'); zlabel('mode 3');
        end
        
        function [I, patch_names] = plot_control_neurons(self, to_sort)
            % Plots 'learned' control neurons from the adaptive dmdc
            % algorithm
            figure
            dat = self.patch_control_neurons;
            patch_names = self.patch_DMD_labels;
            patch_names = patch_names(cellfun(@(x)~isempty(x),patch_names));
            
            if to_sort
                [patch_names, I] = sort(patch_names);
                imagesc(dat(:,I))
            else
                I = [];
                imagesc(dat)
            end
            
            xticks(1:length(patch_names));
            xticklabels(patch_names);
            title('Control neurons for the analyzed behavioral patches')
            
        end
        
        function plot_clustered_control_neurons(self, patch_names)
            if ~exist('patch_names','var')
                to_use_patch_names = false;
            else
                to_use_patch_names = true;
            end
                        
            figure;
            imagesc(self.patch_control_idx')
            title('kmeans clustering results')
            if to_use_patch_names
                xticks(1:length(patch_names));
                xticklabels(patch_names);
            end
        end
    end
    
    methods (Access = private)
        
        function DMD_one_patch(self, index, dat_indices, label)
            % Creates a DMDplotter object for a single patch
            
            if length(dat_indices) <= self.min_patch_size
                warning('Too short (%d<=%d; skipping this patch',...
                    length(dat_indices), self.min_patch_size)
                self.patch_DMD_labels{index} = '';
                return
            else
                dat = self.dat(:,dat_indices);
            end
            
            if self.use_WindowDmd
                this_settings = self.patch_settings;
                this_settings.tspan = dat_indices*self.dt;
                self.patch_DMD_obj{index} = ...
                    WindowDmd(dat, this_settings);
            else
                self.patch_DMD_obj{index} = ...
                    DMDplotter(dat, self.plotter_settings);
            end
                
            self.patch_DMD_labels{index} = label;
        end
        
        function calc_AdaptiveDmdc(self, index, dmdc_settings)
            
            this_dat = self.patch_DMD_obj{index}.dat;
%             [ u_indices, ~, ~ ] = ...
%                 AdaptiveDmdc(this_dat, dmdc_settings);
%             [ u_indices, ~, ~, ~, error_outliers ] = ...
            ad_obj = AdaptiveDmdc(this_dat, dmdc_settings);
            self.AdaptiveDmdc_objects{index} = ad_obj;
            
            if isempty(ad_obj.error_outliers)
                self.patch_control_neurons = [self.patch_control_neurons,...
                    ad_obj.u_indices];
            else
                self.patch_control_neurons = [self.patch_control_neurons,...
                    ad_obj.error_outliers];
            end
        end
        
        function similarity = dynamic_similarity_centroids(self,...
                A, B, iterations, mat_power, label, noise)
            % Calculates matrix similarity using dynamics: multiply each
            % matrix by a noisy centroid of the relevant labeled data
            
            %Generate random vector
            x = zeros([size(A,2), iterations]);
            centroid = self.same_label_struct.centroids(label);
            for i=1:size(A,2)
                x(:,i) = centroid + noise*rand(size(centroid));
            end
            
            similarity = dynamic_similarity(...
                A, B, iterations, mat_power, user_vector);
        end
        
        function similarity = dynamic_similarity_functions(self,...
                A_func, B_func, iterations, label, noise)
            % Calculates data reconstruction similarity using dynamics: 
            % A_func is all dynamics matrices (cell array) to be averaged;
            % B is a single function that will use a randomly
            % generated start point (noise added to average start point of
            % this label)
            
            x0_mean = self.same_label_struct.starts(label);
            similarity = 0;
            for i=1:iterations
                %Generate random vector
                x0 = x0_mean + noise*rand(size(x0_mean));

                % Plug into all dynamics for this label and average the 
                % results
                all_vector = cell2mat(...
                    cellfun(@(f,x) f(x), A_func,...
                    repmat({x0},size(A_func)),...
                    'UniformOutput',false));
                all_vector_mean = mean(all_vector,2);
                
                % Get the (reconstructed) data for the same initial point
                this_vector = B_func(x0);
                
                % Use a cos() for determining similarity
                similarity = similarity + ...
                    real(dot(this_vector, all_vector_mean)) / ...
                    (norm(this_vector)*norm(all_vector_mean));
            end
            similarity = similarity / iterations;
        end
        
        function func_array = calc_DMD_func_array(self, ...
                label, which_patch, t_compare)
            % Calculates the array of reconstruction functions that
            % correspond to a given label
            this_label_indices = self.same_label_struct.indices(label);
            if isnumeric(which_patch)
                %Then only do a single patch
                %   Note: only works if we have multiple DMD A matrices for
                %   a single patch, i.e. the WindowDmd option has been used
                this_label_indices = this_label_indices(which_patch);
            end
            if isempty(this_label_indices)
                error('Found no states with the label %s; aborting', label)
            end
            func_array = {};
            within_patch_tspans = [];
            for index = this_label_indices'
                % Dictionary of all the WindowDmd objects
                % Lots of windows in each patch, each with their own matrix
                % of dynamics
                this_patch_DMDplotters = ...
                    self.patch_DMD_obj{index}.DMDplotter_all;
                
                for key = this_patch_DMDplotters.keys
                    % The actual A matrix for each of these objects is in
                    % an svd projected basis, so we should reconstruct it
                    % from phi and omega (the initial condition is
                    % randomized below)
                    this_DMDplotter = this_patch_DMDplotters(key{1});
                    func_array = [func_array ...
                        {@(x0) ...
                        this_DMDplotter.get_reconstructed_endpoint(...
                        [], t_compare, x0) }]; %#ok<AGROW>
                    within_patch_tspans = [within_patch_tspans ...
                        this_DMDplotter.tspan]; %#ok<AGROW>
                end
            end
            self.dynamic_matrices_dict(label) = func_array;
            self.same_label_struct.within_patch_tspans(label) = ...
                within_patch_tspans;
        end
        
        function [patch_names, I] = sort_patch_names(self)
            patch_names = self.patch_DMD_labels;
            patch_names = patch_names(cellfun(@(x)~isempty(x),patch_names));
            
            [patch_names, I] = sort(patch_names);
        end
        
    end
    
    methods (Static)
        function similarity = pairwise_similarity(A, B,...
                                                  calc_A_svd, calc_B_svd)
            %Calculates pairwise similarity between matrices A and B
            %   Alternatively, pass the svd S matrix for one or both of
            %   them
            if ~exist('calc_A_svd','var') || calc_A_svd
                [~,A_S,~] = svd(A);
                A_S = diag(A_S);
            else
                A_S = A;
            end
            if ~exist('calc_B_svd','var') || calc_B_svd
                [~,B_S,~] = svd(B);
                B_S = diag(B_S);
            else
                B_S = B;
            end
            similarity = dot(A_S, B_S)/(norm(A_S)*norm(B_S));
        end
        
        function similarity = dynamic_similarity(...
                A, B, iterations, mat_power, user_vector)
            % Calculates matrix similarity using dynamics: multiply each
            % matrix by the same random vector a number of times and 
            % compare the resulting vectors (inner product)
%             if ~exist('use_negative','var')
%                 use_negative = false;
%             end
            
            A_n = A^mat_power;
            B_n = B^mat_power;
            % Use a single matrix for the random vectors (columns)
            if ~exist('user_vector', 'var')
                x = rand([size(A,2), iterations]);
            else
                x = user_vector;
            end
%             if use_negative
%                 x = 2*x-1;
%             end
            A_n_times_x = A_n*x;
            B_n_times_x = B_n*x;
            similarity = 0;
            for i=1:iterations
                Ax = A_n_times_x(:,i);
                Bx = B_n_times_x(:,i);
                similarity = similarity + ...
                    abs(dot(Ax,Bx)) / (norm(Ax)*norm(Bx));
            end
            similarity = similarity / iterations;
        end
    end
    
end

