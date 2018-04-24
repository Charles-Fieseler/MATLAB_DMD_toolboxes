classdef HierarchicalDmd < handle & AbstractDmd
    %% HierarchicalDmd
    % Performs Dynamic Mode Decomposition on hierarchically separated
    % subsets of data (uses C elegans neuron categories to get subsets and
    % PatchDmd object to analyze each subset)
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
        % Settings for each PatchDmd object (one per subset)
        PatchDmd_settings = struct(...
            'patch_settings',...
            struct('window_size', 20, 'window_step', 1,... 
            'plotter_settings',...
            struct('use_optdmd', true, 'model_order', 10) ) )
        
        % Type of hierarchical breakdown
        use_celegans_classes = true
        use_derivatives = false
%         class_names = {'sensory', 'inter', 'motor'};
        class_type2index = containers.Map(...
            {'NaN', 'sensory', 'inter', 'motor'},...
            {0, 1, 2, 3})
        class_ids
    end
    
    properties (SetAccess=private, Hidden=true) 
        % hierarchy variables
%         cell_ID
        full_dat
        dat_classes
        
        % original data variables
        patch_labels
        patch_vector
        patch_starts
        
        tspan_full
        patch_DMD_labels

        cell_id_struct
        cell_id_unique
    end
    
    properties (SetAccess=private) 
        patch_DMD_obj
    end
    
    methods
        
        function self = HierarchicalDmd(file_or_dat, settings)
            
            %% Initialize and check
            self.import_settings_to_self(settings);
            self.patch_DMD_obj = struct();
            self.dat_classes = struct();
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
            
            %% Get hierarchy then do DMD
            self.calc_hierarchy();
            self.DMD_all_subsets();
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
            self.full_dat = dat;
            options = struct('verbose',false);
            % Calcium imaging magnitudes or derivatives
            if self.use_derivatives
                self.raw = dat.tracesDif';
            else
                self.raw = dat.traces';
            end
%             self.tspan_full = dat.timeVectorSeconds;
%             self.dt = self.tspan_full(2) - self.tspan_full(1);
            % Patch information (hand-labeled)
            self.patch_vector = dat.(state_strs{1});
            self.patch_labels = dat.(state_strs{2});
            self.patch_starts = find(diff(self.patch_vector));
            % Cell information
            num_neurons = size(self.raw,1);
            self.cell_id_unique = repmat({''}, [num_neurons 1]);
            for id = ID_strs'
                this_id_array = dat.(id{1});
                id_struct.(id{1}) = this_id_array;
                for iCell=1:num_neurons
                    if isempty(self.cell_id_unique{iCell})
                        self.cell_id_unique{iCell} = this_id_array{iCell};
                    elseif strcmp(self.cell_id_unique{iCell},this_id_array{iCell})
                        continue
                    elseif ~isempty(this_id_array{iCell}) &&...
                            ~any(isnan(self.cell_id_unique{iCell}))
                        % The entry is different from the previously saved
                        % one, but their types might be the same
                        t_old = CEinfo(self.cell_id_unique{iCell},options);
                        t_new = CEinfo(this_id_array{iCell},options);
                        if ~strcmp(t_old.wormatlasData.type,...
                                t_new.wormatlasData.type)
                            self.cell_id_unique{iCell} = NaN;
                        else
                            % Leave it be
                        end
                    end
                end
            end
            self.cell_id_struct = id_struct;
            
            % Cell array for the actual analysis objects
%             self.patch_DMD_obj = cell(length(self.patch_starts), 1);
            self.patch_DMD_labels = cell(length(self.patch_starts), 1);
        end
        
        function calc_hierarchy(self)
            % Calculates the hierarchy that splits the data
            %   If use_celegans_classes is true, then splits into 3 classes:
            %       motor
            %       inter
            %       sensory
            
            assert(self.use_celegans_classes,...
                'Only C elegans hierarchy currently implemented')
            
            options = struct('verbose',false);
            self.class_ids = zeros(self.sz(1), 1);
            for iCell=1:self.sz(1)
                this_cell_name = self.cell_id_unique{iCell};
                if isempty(this_cell_name) || any(isnan(this_cell_name))
                    continue
                else
                    tmp = CEinfo(this_cell_name, options);
                    this_cell_type = tmp.wormatlasData.type;
                    if isempty(this_cell_type)
                        % Then it is ambiguous in the connectome
                        self.class_ids(iCell) = 0;
                    else
                        self.class_ids(iCell) = ...
                            self.class_type2index(this_cell_type);
                    end
                end
            end
            
            % Split the data up by unique category
            for key = self.class_type2index.keys
                this_ind = self.class_ids==self.class_type2index(key{1});                
                % Struct with the rest of the metadata
                this_full_dat = self.full_dat;
                this_full_dat.traces = self.dat(this_ind,:)';
                this_full_dat.ID = this_full_dat.ID(this_ind);
                this_full_dat.ID2 = this_full_dat.ID2(this_ind);
                this_full_dat.ID3 = this_full_dat.ID3(this_ind);
                self.dat_classes.(key{1}) = this_full_dat;
            end
            
        end
        
        function DMD_all_subsets(self)
            % Uses PatchDmd object to analyze the traces of each class of
            % neurons
            
            for key = self.class_type2index.keys
                k = key{1};
                self.patch_DMD_obj.(k) = PatchDmd(...
                    self.dat_classes.(k),...
                    self.PatchDmd_settings);
            end
        end
    end
    
    methods % processing
        
        function postprocess(self)
            % Calculate various class properties
            
            for key = self.class_type2index.keys
                k = key{1};
                this_obj = self.patch_DMD_obj.(k);
                this_obj.initialize_all_similarity_objects();
                this_obj.calc_all_label_similarities();
%                 self.patch_DMD_obj(k) = this_obj;
            end
        end
        
        function preprocess(self)
            % Set initial data properties (mostly a stub)
            self.original_sz = size(self.raw);
            self.dat = self.raw;
            self.sz = size(self.dat);
        end
    end
    
    methods % plotting
        function plot_same_label_similarities(self,...
                which_class, which_behavior, plot_mode)
            % Plots similarities
            
            this_obj = self.patch_DMD_obj.(which_class);
            if ~strcmp(which_behavior,'all')
                this_obj.plot_same_label_similarity(which_behavior, plot_mode);
                title( sprintf(...
                    '%s dynamics from %s-class neurons',...
                    which_behavior, which_class) )
            else
                this_obj.calc_all_label_similarities(true);
            end
        end
    end
end