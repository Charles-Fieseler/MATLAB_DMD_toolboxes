classdef MultiResDmd < AbstractDmd
    %Multi-resolution DMD: DMD at multiple timescales
    %   Performs DMD, then subtracts the slow modes, cuts the remaining
    %   data into 2 halves; repeat
    %
    %
    % INPUTS
    %   file_or_dat - filename or data matrix; could also be mrDMD object
    %   settings - Struct of settings
    %
    % OUTPUTS -
    %   MultiResDmd object - object that does DMD on windows with many
    %   plotting options
    %
    % EXAMPLES
    %
    %   EXAMPLE1
    %
    %
    % Dependencies
    %   Other m-files required: (updated on 29-Nov-2017)
    %             MATLAB (version 9.2)
    %             Statistics and Machine Learning Toolbox (version 11.1)
    %             v2struct.m
    %             dmd.m
    %             PlotterDmd.m
    %             AbstractDmd.m
    %             plotSVD_Rice.m
    %             prox_func.m
    %
    %   See also: OTHER_FUNCTION_NAME
    %
    %
    % Author: Charles Fieseler
    % University of Washington, Dept. of Physics
    % Email address: charles.fieseler@gmail.com
    % Website: coming soon
    % Created: 29-Nov-2017
    %========================================
    
    properties (SetAccess=private, Hidden=true)
        %DMD outputs, in containers.map objects
        omega_all
        phi_all
        coeff_all
        dat_layers
    end
    
    properties (SetAccess={?SettingsImportableFromStruct})
        okay_collapse_dat     = true %If trailing data points can be collapsed to make the binning work
        %If using the PlotterDmd object
        use_plotter_obj       = true
        %User processing settings
        num_layers           = 1
        cutoff_func          = @abs
        cutoff              = 0.1
        all_real_subtraction  = false
        %If the data is input as a struct
        import_field_name     = '';
    end
    
    methods
        %Constructor
        function self = MultiResDmd(file_or_dat, settings)
            %% Initialize with defaults
            self.import_settings_to_self(settings);
            
            % Initialize the DMD object containers
            self.dat_layers = containers.Map();
            self.omega_all = containers.Map();
            self.phi_all = containers.Map();
            self.coeff_all = containers.Map();
            %==========================================================================
            
            %% Import data and preprocess
            if ischar(file_or_dat)
                self.filename = file_or_dat;
                self.raw = importdata(file_or_dat);
                if isstruct(self.raw)
                    self.raw = self.raw.(self.import_field_name);
                end
            elseif isnumeric(file_or_dat)
                self.filename = '';
                self.raw = file_or_dat;
            elseif isstruct(file_or_dat) && ...
                    ~isempty(self.import_field_name)
                self.filename = file_or_dat;
                tmp_struct = importdata(file_or_dat);
                self.raw = tmp_struct.(self.import_field_name);
            else
                error("First argument should either be a filename, matrix, or struct")
            end
            self.preprocess();
            %==========================================================================
            
            %% Do dmd for num_layers number of layers
            self.DMD_all_layers();
            %==========================================================================

        end
        
        %Do DMD with two different data splittings
        function DMD_all_layers(self, dat_key)
            %Does DMD layer by layer and saves the results
            %   Can start with the key of a given layer
            if ~exist('dat_key','var')
                dat_key = [];
            end
            for jL = 1:self.num_layers
                if self.verbose
                    fprintf('Processing layer %d\n',jL)
                end
                layer_indices = self.get_layer_indices(jL);
                for jB = 1:(2^(jL-1))
                    self.DMD_one_bin(jL, jB, layer_indices(:,jB), dat_key)
                end
                %After each layer, save the data
                self.dat_layers(self.vec2key([jL,-1])) = ...
                    self.dat;
            end
        end
        
        %Plotter functions
        function plot_reconstruction(self, num_modes, which_layers)
            %Plots a reconstruction of the original data using the
            %specified modes from the specified layers
            if isscalar(which_layers)
                vec = [num_modes,which_layers];
                key = self.vec2key(vec);
                which_layers = 1:which_layers;
            else
                vec = [num_modes,which_layers(end)];
                key = self.vec2key(vec);
            end
            assert(max(which_layers)<=self.num_layers,...
                'Cannot reconstruct using more layers than exist')
            
            if ~self.approx_all.isKey(key)
                self.get_reconstruction(num_modes, key, which_layers);
            end
            dat_approx = self.approx_all(key);
            
            figure
            subplot(2,1,1)
            jDat = self.dat_layers('0 -1');
            imagesc(jDat(1:self.original_sz(1),:));
            title('Original data')
            ylabel('Neuron number')
            xlabel('Time')
            colorbar;
            lim = caxis; %Get colorbar limits
            
            subplot(2,1,2)
            imagesc(dat_approx(1:self.original_sz(1),:));
            title(sprintf(...
                'Reconstruction with %d mode(s) and %d layer(s)',...
                vec(1),vec(2)));
            colorbar;
            caxis(lim);
        end
        
        function plot_power_spectrums(self, layer, use_coeff, use_real_omega)
            %Plots the power spectra of ALL the bins
            if ~exist('layer','var')
                layer = 1;
            end
            if ~exist('use_coeff','var')
                use_coeff = false;
            end
            if ~exist('use_real_omega','var')
                use_real_omega = false;
            end
            
            figure
            hold on
            for j=1:self.PlotterDmd_all.length()
                key = self.vec2key([layer, j]);
                if self.PlotterDmd_all.isKey(key)
                    obj = self.PlotterDmd_all(key);
                    obj.plot_power_spectrum(use_coeff, use_real_omega);
                else
                    continue;
                end
            end
        end
        
        function plot_layer(self, key, neur_num)
            %Plots all neurons for a single layer after DMD subtraction
            if ~exist('key','var') || isempty(key)
                %This refers to the original data
                key = '0 -1';
                vec = [0,-1];
            elseif isnumeric(key)
                vec = key;
                key = self.vec2key(key);
            end
            
            figure('defaultAxesFontSize',14);
            thisDat = real(self.dat_layers(key));
            tStart = 1;
            if ~self.augmentData
                thisDat = thisDat(:,tStart:end);
%                 thisDat = real(self.dat(:,tStart:end));
            else
                thisDat = thisDat(1:self.original_sz(1),tStart:end);
%                 thisDat = real(self.dat(1:self.original_sz(1),tStart:end));
            end
            if ~exist('neur_num','var')
                %By default, show all neurons
                imagesc(thisDat);
                ylabel('Neuron number')
                colorbar;
            else
                %Here, only show one neuron
                plot(thisDat(neur_num,:));
                ylabel('Calcium amplitude')
            end
            title(sprintf('Data for layer %d and bin %d',vec(1),vec(2)))
            xlabel('Time')
        end
        
        function thisDat = plot_neuron_foreground(self,...
                neur_num, ~, layer_num, show_raw)
            %Plots a single neuron vs. (optionally) the raw data
            if ~exist('show_raw','var')
                show_raw = false;
            end
            figure;
            key = self.vec2key([layer_num, -1]);
            thisDat = self.dat_layers(key);
            plot(real(thisDat(neur_num,:)))
            title(sprintf(...
                'Neuron foreground %d filtered through %d layer(s) ',...
                neur_num, layer_num))
            xlabel('Time (frames)')
            ylabel('Calcium imaging amplitude')
            
            if show_raw
                self.plot_layer([],neur_num);
                title(sprintf('Neuron %d (raw data)',...
                    neur_num))
            end
        end
        
        function thisDat = plot_neuron_background(self,...
                neur_num, ~, layer_num, show_raw)
            %Plots a reconstruction of the original data using the
            %specified modes from the specified layers
            %   Note: same as plot_reconstruction, but this is only a
            %   single neuron
%             if isscalar(which_layers)
%                 vec = [num_modes,which_layers];
%                 key = self.vec2key(vec);
%                 which_layers = 1:which_layers;
%             else
%                 vec = [num_modes,which_layers(end)];
%                 key = self.vec2key(vec);
%             end
            warning('Need to check the bin number here')
            vec = [layer_num, 1];
            key = self.vec2key(vec);
            
            XdmdSlow = self.get_slow_DMD([], [], key);
            if ~self.approx_all.isKey(key)
%                 self.get_reconstruction(num_modes, key, layer_num);
            end
            %dat_approx = self.approx_all(key);
            
            figure
            thisDat = XdmdSlow(neur_num,:);
            plot(real(thisDat))
            title(sprintf('Neuron background %d filtered through %d layer(s) ',...
                neur_num, layer_num))
            xlabel('Time (frames)')
            ylabel('Calcium imaging amplitude')
            
            if show_raw
                self.plot_layer([],neur_num);
                title(sprintf('Neuron %d (raw data)',...
                    neur_num))
            end
        end
        
        %Data processing
        function preprocess(self)
            % Uses the abstract class to handle most of the preprocessing
            preprocess@AbstractDmd(self);
            
            %Make sure data is binnable
            if mod(self.sz(2),2^(self.num_layers-1))~=0
                if ~self.okay_collapse_dat
                    error('Each data bin should have a whole number of data points')
                else
                    warning('Throwing away extra data points')
                    bins = 2^(self.num_layers-1);
                    keep = 1:(floor(self.sz(2)/bins)*bins);
                    self.raw = self.raw(:,keep);
                    self.sz = size(self.raw);
                end
            end
            
            %Save original (preprocessed) data
            self.dat_layers('0 -1') = self.dat;
        end
    end
    
    methods (Access = private)
        
        %Multi-resolution DMD without overlapping bins
        function layer_ind = get_layer_indices(self, layer)
            %Gets the layer indices by dividing in half
            bins = 2^(layer - 1);
            layer_sz = self.sz(2)/bins;
            layer_ind = zeros(layer_sz,bins);
            for j=1:bins
                iStart = (j-1)*layer_sz + 1;
                iEnd = iStart + layer_sz - 1;
                layer_ind(:,j) = (iStart:iEnd)';
            end
        end
        
        %Functions for both types of DMD
        function DMD_one_bin(self, layer, index, dat_indices, dat_key)
            %Does DMD on a single file and saves the results
            vec = [layer, index]; %Initial layer is 1; only a single time bin
            key = self.vec2key(vec);
            if ~exist('dat_key','var') || isempty(dat_key)
                mydat = self.dat(:,dat_indices);
            else
                datLayer = self.dat_layers(dat_key);
                mydat = datLayer(:,dat_indices);
            end
            if ~self.use_plotter_obj
                self.subtract_slow_DMD(mydat, dat_indices, []);
            else
                self.PlotterDmd_all(key) = ...
                    PlotterDmd(mydat, self.plotterSet);
                self.subtract_slow_DMD(mydat, dat_indices, key);
            end
        end
        
        function subtract_slow_DMD(self, mydat, dat_indices, key)
            %Get the slow modes and subtract them out
            %   This saves the DMD data as well
            
            XdmdSlow = self.get_slow_DMD(mydat, dat_indices, key);
            
            if ~self.all_real_subtraction
                self.dat(:,dat_indices) = mydat - XdmdSlow;
            else
                %Avoids the problem of subtracting a complex component from
                %the data in order to keep it completely real AND positive
                XdmdFast = mydat - abs(XdmdSlow);

                %This is almost the foreground, but there will probably be negative values
                %in this matrix, which will to be removed
                residuals = min(XdmdFast,0);

%                 XdmdSlow = residuals + abs(XdmdSlow); %Background
%                 XdmdFast = XdmdFast - residuals; %Foreground
                self.dat(:,dat_indices) = XdmdFast - residuals;
            end
            
        end
        
        function XdmdSlow = get_slow_DMD(self, mydat, dat_indices, key)
            %Gets the slow modes
            
            %Do DMD or get it from the plotter object
            if ~self.use_plotter_obj
                [coeff, omega, phi ] = ...
                    DMD( mydat, self.dt, self.modelOrder, self.dmdPercent);
            else
                obj = self.PlotterDmd_all(key);
                coeff = obj.coeff_sort;
                omega = obj.Omega_sort;
                phi = obj.Phi_sort;
            end
            
            %Set the timespan
            if ~isempty(dat_indices) 
                tspan = dat_indices*self.dt;
            else
                tspan = (1:obj.sz(2))*self.dt;
            end
            if self.t0EachBin
                %Set first entry to 0
                tspan = tspan - tspan(1);
            end
            
            assert(~isempty(omega), 'No modes found')
            %Separates modes with frequency near origin
            vec = self.key2vec(key);
            if vec(1)~=self.num_layers || vec(1)==1
                %The first layer should always be subtracted
                slowInd = self.cutoff_func(omega)<self.cutoff;
            else
                %then we're on the last layer and we keep all modes
                slowInd = ones(size(omega));
            end
            if isempty(find(slowInd, 1))
                %If there are no 'slow' modes, pick the slowest one
                [~, slowInd] = min(abs(omega));
            end
            omega = omega(slowInd);
            phi = phi(:,slowInd);
            coeff = coeff(slowInd);

            for jT = length(tspan):-1:1
                XmodesSlow(:,jT) = coeff.*exp(omega*tspan(jT));
            end
            XdmdSlow = phi*XmodesSlow;

            %Save dmd mode data
            self.omega_all(key) = omega;
            self.phi_all(key) = phi;
            self.coeff_all(key) = coeff;
        end
        
        %Functions for approximation reconstruction
        function get_reconstruction(self, num_modes, key, which_layers)
            
            dat_approx = zeros(self.sz);
            for jL=which_layers
                thisLayer = zeros(self.sz);
                layerInd = self.get_layer_indices(jL);
                tspan = layerInd*self.dt;
                if self.t0EachBin
                    tspan = tspan - tspan(1) + self.dt;
                end
                numBins = 2^(jL-1);
                for jB = 1:numBins
                    thisDMD = self.PlotterDmd_all(self.vec2key([jL, jB]));
                    thisLayer(:, layerInd(:, jB)) = ...
                        thisDMD.get_reconstruction(num_modes, tspan);
                end
                dat_approx = dat_approx + thisLayer;
            end
            dat_approx = real(dat_approx);
            self.approx_all(key) = dat_approx;
        end
    end
    
end

