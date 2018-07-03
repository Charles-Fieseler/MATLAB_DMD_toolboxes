classdef WindowDmd < handle & AbstractDmd
    %WindowDmd - does DMD on many different windows which are overlapping
    %by default
    %
    %
    % INPUTS
    %   file_or_dat - filename or data matrix; could also be MultiResDmd object
    %   settings - Struct of settings
    %
    % OUTPUTS -
    %   WindowDmd object - object that does DMD on windows with many
    %   plotting options
    %
    % EXAMPLES
    %
    %   EXAMPLE1
    %
    %
    %   EXAMPLE2
    %
    %
    % Dependencies
    %   Other m-files required: (updated on 29-Nov-2017)
    %             MATLAB (version 9.2)
    %             Statistics and Machine Learning Toolbox (version 11.1)
    %             v2struct.m
    %             DMD.m
    %             PlotterDmd.m
    %             AbstractDmd.m
    %             plotSVD_Rice.m
    %             prox_func.m
    %
    %   See also: OTHER_FUNCTION_NAME
    %
    %
    %
    % Author: Charles Fieseler
    % University of Washington, Dept. of Physics
    % Email address: charles.fieseler@gmail.com
    % Website: coming soon
    % Created: 29-Nov-2017
    %========================================
    
    
    properties (SetAccess={?SettingsImportableFromStruct}, Hidden=false)
        window_size          = 100
        window_step          = 10
        %If importing from mrdat object
        dat_key             = '0 -1'
        %Library settings
        lib_use_coeff_sign     = false
        lib_use_abs           = false
        sort_func            = @(x,phi,xbar,sigma)x > xbar + sigma
    end
    
    properties (SetAccess=private, Hidden=true)
        %Library of modes with power >1sigma above the average in a window
        dat_library
        dat_lib_ind
        dat_V %for svd
        dat_U
        dat_S
        dat_idx
        dat_centroids
        num_clusters
        %Initialized in preprocessing
%         num_clusters
        window_ind
        window_normalization
    end
    
    methods
        
        function self = WindowDmd(file_or_dat, settings)
            %% Initialize and check
            self.import_settings_to_self(settings);
            
            if self.lib_use_abs && self.lib_use_coeff_sign
                warning('Cannot use both lib_use_abs and lib_use_coeff_sign')
            end
            %==========================================================================
            
            %% Import data and preprocess
            if ischar(file_or_dat)
                self.filename = file_or_dat;
                self.raw = importdata(file_or_dat);
                self.preprocess(true);
            elseif isnumeric(file_or_dat)
                self.filename = '';
                self.raw = file_or_dat;
                self.preprocess(true);
            elseif isa(file_or_dat,'MultiResDmd')
                self.copy_from_MultiResDmd(file_or_dat);
                self.preprocess(false);
            else
                error('Unrecognized data type')
            end
            %==========================================================================
            
            %% Do dmd for all windows
            self.DMD_all_windows();
            %==========================================================================
            
        end
        
        function cluster_library(self, num_modes, k, cluster_mode)
            %Uses kmeans clustering on the library of 'interesting' modes
            
            warning('Clusters are not yet quantitatively consistent; use for preliminary work only')
            
            if ~exist('cluster_mode','var')
                cluster_mode = 'kmeans';
            end
            if isempty(self.dat_V)
                [U, S, V] = svd(self.dat_library);
                self.dat_V = V;
                self.dat_U = U;
                self.dat_S = S;
            else
                V = self.dat_V;
            end
            if isnumeric(num_modes)
                V = real(V(:,1:num_modes));
            else
                V = real(V);
            end
            
            if ~exist('k','var') || isempty(k)
                rng('default');  % For reproducibility
                eva = evalclusters(...
                    V,cluster_mode,'CalinskiHarabasz','KList',1:6);
                k = eva.OptimalK;
                if self.verbose
                    fprintf('Clustering using the optimal value k=%d\n',k);
                end
            elseif self.verbose
                fprintf('Clustering with k=%d using %s\n',...
                    k, cluster_mode);
            end
            
            switch cluster_mode
                case 'kmeans'
                    opts = statset('MaxIter',1000);
                    [self.dat_idx, self.dat_centroids] = ...
                        kmeans(V, k, 'Options', opts);
                case 'gmm'
                    gmmobj = fitgmdist(V, k);
                    idx = cluster(gmmobj, V);
                    
                    self.dat_centroids = zeros(k,num_modes);
                    for j=1:k
                        thisCluster = V(idx == j,:); % |1| for cluster 1 membership
                        self.dat_centroids(j,:) = mean(thisCluster,1);
                    end
                    self.dat_idx = idx;
                    
                otherwise
                    error('Unrecognized clustering algorithm')
            end
            
            self.num_clusters = k;
        end
        
        function preprocess(self, full_preprocess)
            % Uses the abstract class to handle most of the preprocessing
            if full_preprocess
                preprocess@AbstractDmd(self);
            end
            
            self.set_default_windows();
            self.set_window_indices();
            self.set_window_normalization();
        end
        
    end
    
    methods % Plotting
        
        function plot_all_power_spectra(self, use_coeff, use_real_omega)
            % Plots the power spectra of ALL the windows
            % Input:
            %   use_coeff (false) - Uses the sign of the real part to plot
            %   use_real_omega (false) - Uses only the real part of omega
            if ~exist('use_coeff','var')
                use_coeff = false;
            end
            if ~exist('use_real_omega','var')
                use_real_omega = false;
            end
            
            figure
            hold on
            for j=1:self.PlotterDmd_all.length()
                key = self.vec2key(j);
                if self.PlotterDmd_all.isKey(key)
                    obj = self.PlotterDmd_all(key);
                    obj.plot_power_spectrum(use_coeff, use_real_omega);
                else
                    continue;
                end
            end
            title('Power spectrum for all windows')
        end
        
        function fig = plot_power_and_data(self,...
                index, fig, use_coeff_sign, use_real_omega, ...
                subplot_func)
            %Plots the power spectra of a single window, with gui
            if ~exist('index','var')
                index = 1;
            end
            if ~exist('fig','var') || isempty(fig)
                fig = figure;
            end
            if ~exist('use_coeff_sign','var')
                use_coeff_sign = false;
            end
            if ~exist('use_real_omega','var')
                use_real_omega = false;
            end
            if ~exist('subplot_func','var')
                subplot_func = @plot_power_spectrum;
            end
            
            subplot(2,1,1)
            self.plot_window(index, use_coeff_sign, use_real_omega, ...
                subplot_func);
            
            subplot(2,1,2)
            key = self.vec2key(index);
            dmdobj = self.PlotterDmd_all(key);
            if isempty(self.dat_idx)
                subplot_func(dmdobj, use_coeff_sign, use_real_omega);
            else
                categories = zeros(size(dmdobj.coeff));
                %Get 'interesting' modes for this window
                %                 thisLibInd = find(self.dat_lib_ind(:,1)==index); %Indices in the library
                %                 thisLibCat = self.dat_idx(thisLibInd); %Categories in the library
                %
                %                 thisWindInd = self.dat_lib_ind(thisLibInd,2); %Indices in this window
                for jC=1:self.num_clusters
                    thisWindInd = self.get_modes_from_cluster(index,jC);
                    categories(thisWindInd) = jC;
                end
                %                 for j = 1:length(thisWindInd)
                %                     categories(thisWindInd(j)) = thisLibCat(j);
                %                 end
                
                dmdobj.plot_power_spectrum(...
                    use_coeff_sign, use_real_omega, categories);
            end
            title(sprintf('Power spectrum for window %d',index))
        end
        
        function plot_window(self, index,...
                use_coeff_sign, use_real_omega, subplot_func)
            if ~exist('use_coeff_sign','var')
                use_coeff_sign = false;
            end
            if ~exist('use_real_omega','var')
                use_real_omega = false;
            end
            if ~exist('subplot_func','var')
                subplot_func = @plot_power_spectrum;
            end
            
            assert( index>0 && index<=self.num_clusters,...
                'Must be valid window index')
            
            self.plot_original_data(false);
            hold on
            start_x = self.window_ind(1,index);
            end_x = self.window_ind(end,index)+1;
            plot([start_x, start_x], [1, self.original_sz(1)],...
                'k','LineWidth',2)
            plot([end_x, end_x], [1, self.original_sz(1)],...
                'k','LineWidth',2)
            %For interactivity
            bar_x = linspace(1,self.sz(2),self.num_clusters);
            plot(bar_x, ones(size(bar_x)),'ok','ButtonDownFcn',...
                @(src,evt) self.plot_window_callback(...
                src, evt, use_coeff_sign, use_real_omega, subplot_func),...
                'LineWidth',2)
            ylabel('Neuron number')
            colorbar;
            title(sprintf('Data for window %d',index))
            xlabel('Time')
        end
        
        function plot3d_library_svd(self)
            %Plots a simple 3d svd view of the 'interesting' modes
            
            if isempty(self.dat_V)
                [~, ~, V] = svd(self.dat_library);
                self.dat_V = V;
            else
                V = self.dat_V;
            end
            
            figure
            if isempty(self.dat_idx)
                %Then we have no clusters
                plot3(V(:,1), V(:,2), V(:,3), 'o')
            else
                hold on
                leg = cell(self.num_clusters,1);
                for j=1:self.num_clusters
                    jInd = self.dat_idx==j;
                    plot3(V(jInd,1), V(jInd,2), V(jInd,3), 'o')
                    leg{j} = sprintf('cluster %d',j);
                end
                legend(leg)
            end
            
            title(sprintf('%d modes with a cutoff of 1 sigma',...
                size(V,1)))
            xlabel('First mode')
            ylabel('Second mode')
            zlabel('Third mode')
        end
        
        function plot_library_svd(self)
            %Uses external SVD function to plot basic information
            
            plotSVD_Rice(self.dat_library);
        end
        
        function plot_power_classes(self,...
                use_coeff, use_power_decay, to_normalize_classes, ...
                use_color_bar, tstart)
            %Plots the total power in each category of clustered
            %'interesting' DMD modes
            assert(self.num_clusters>0,...
                'Data has not been clustered yet; run self.cluster_library(...)')
            if ~exist('use_coeff','var')
                use_coeff = true;
            end
            if ~exist('use_power_decay','var')
                use_power_decay = true;
            end
            if ~exist('to_normalize_classes','var')
                to_normalize_classes = false;
            end
            if ~exist('use_color_bar','var')
                use_color_bar = false;
            end
            if ~exist('tstart','var')
                tstart = 1;
            end
            
            all_power = zeros(self.num_clusters,self.sz(2));
            figure
            subplot(2,1,1)
            self.plot_original_data(use_color_bar, tstart);
            
            subplot(2,1,2)
            hold on
            titleStr = '';
            leg = cell(self.num_clusters,1);
            
            for jC=1:self.num_clusters
                for jW=1:self.num_clusters
                    key = self.vec2key(jW);
                    dmdobj = self.PlotterDmd_all(key);
                    
                    thisWindInd = self.get_modes_from_cluster(jW, jC);
                    thisInd = self.window_ind(:, jW);
                    
                    %Sum over both the spatial extent of the mode and the
                    %entire frequency spectrum to get a general "how much
                    %power is in the class of mode"
                    if ~isempty(thisWindInd)
                        if use_coeff
                            if ~dmdobj.useScaling
                                %Usually, the modes have a norm equal to 1, so
                                %the coefficients should be used to get a power
                                %estimate
                                theseCoeff = abs(...
                                    dmdobj.coeff_sort(thisWindInd) ).^2;
                            else
                                %Collapse the space dimension
                                % If use_coeff=false, then we come here and get
                                % 1 for each mode, if normalized
                                theseCoeff = sum(abs(...
                                    dmdobj.phi_sort(:,thisWindInd) ).^2, 1);
                            end
                        else
                            %Just count the number of modes
                            titleStr = 'Number of modes in each cluster';
                            theseCoeff = ones(size(thisWindInd));
                        end
                        if ~use_power_decay
                            %Then we can just get a scalar 'power' for the
                            %entire window
                            if isempty(titleStr)
                                titleStr = 'Power in each cluster (constant throughout window)';
                            end
                            thisPowerMat = real(...
                                ones(size(thisInd'))*sum(theseCoeff) );
                        else
                            %Use an exponential to decay/grow throughout
                            %the window
                            if isempty(titleStr)
                                titleStr = 'Power in each cluster (using decay)';
                            end
                            theseOmega = dmdobj.omega_sort(thisWindInd);
                            tspan = thisInd'*self.dt;
                            if self.t0_each_bin
                                tspan = tspan - tspan(1);
                            end
                            thisPowerMat = real(sum(...
                                theseCoeff.*abs(exp(theseOmega.*tspan)).^2,...
                                1 ));
                        end
                    else
                        thisPowerMat = zeros(size(thisInd'));
                    end
                    
                    %Add it all up
                    all_power(jC, thisInd) = ...
                        all_power(jC, thisInd) + ...
                        thisPowerMat;
                end
                leg{jC} = sprintf('Class %d', jC);
                all_power(jC,:) = ...
                    all_power(jC,:) ./ self.window_normalization;
                if to_normalize_classes
                    %Make all of the classes comparable
                    all_power(jC,:) = ...
                        all_power(jC,:) ./ max(all_power(jC,:));
                    ylabel('Power (normalized for EACH cluster)')
                else
                    ylabel('Power')
                end
                plot(all_power(jC,tstart:end),'LineWidth',2)
            end
            set(gca,'xlim',[1 self.sz(2)])
            legend(leg)
            xlabel('Time (frames)')
            title(titleStr)
            
        end
        
        function lim = plot_original_data(self, use_color_bar, tstart)
            %Sizes the data to plot a non-augmented version of the original
            %data
            if ~exist('use_color_bar','var')
                use_color_bar = true;
            end
            if ~exist('tstart','var')
                tstart = 1;
            end
            
            if ~self.augment_data
                mydat = real(self.dat(:,tstart:end));
            else
                mydat = real(self.dat(1:self.original_sz(1),tstart:end));
            end
            imagesc(mydat)
            title('Original data')
            if use_color_bar
                colorbar;
                lim = caxis; %Get colorbar limits
            else
                lim = [];
            end
        end
        
        function plot_reconstruction(self, num_modes, which_clusters)
            %Plots a reconstruction of the data using a weighted average of
            %window DMD modes
            %   Optional: only plot certain clusters, as identified using
            %   self.cluster_library
            
            if ~exist('which_clusters','var')
                which_clusters = 0:self.num_clusters;
            end
            
            key = self.vec2key(which_clusters);
            if ~self.approx_all.isKey(key)
                self.get_reconstruction(num_modes, which_clusters);
            end
            dat_approx = self.approx_all(key);
            
            figure
            subplot(2,1,1)
            lim = self.plot_original_data();
            
            subplot(2,1,2)
            title(sprintf('Data reconstruction from %d clusters', ...
                length(which_clusters)))
            imagesc(dat_approx)
            colorbar;
            caxis(lim);
            
        end
        
        function plot_centroids(self, which_modes)
            %Plots the centroid of the kmeans clusters, if saved
            assert(~isempty(self.dat_centroids),...
                'Data not yet clustered; run self.cluster_library(...)')
            
            %The saved data is in SVD basis, so let's put it back into
            %neuron basis
            sz = size(self.dat_centroids);
            dat_tmp = real(...
                self.dat_U(:,1:sz(2))*(self.dat_centroids'));
            dat_neuron = dat_tmp(1:self.original_sz(1),:);
            for j=1:size(dat_neuron,2)
                dat_neuron(:,j) = ...
                    dat_neuron(:,j)/mean(abs(dat_neuron(:,j)));
            end
            
            figure('defaultAxesFontSize',14);
            hold on
            plot_colors = lines(max(which_modes));
            leg = {};
            for j=unique(which_modes)
                %plot(dat_neuron(:,j), 'LineWidth', 2)
                bar(dat_neuron(:,j))
                leg = [leg sprintf('cluster %d',j)]; %#ok<AGROW>
                %match up the colors by hand
                f = gca();
                %f.Children(1).Color = plot_colors(j,:);
                f.Children(1).FaceColor = plot_colors(j,:);
            end
            legend(leg)
            title('Centroids of the DMD mode clusters')
            xlabel('Neuron')
            ylabel('Amplitude')
        end
        
    end
    
    methods (Access=private)
        
        function DMD_all_windows(self)
            %Does DMD on the data using a sliding window
            %   Always saves the bins using PlotterDmd objects
            
            for j=1:self.num_clusters
                if self.verbose
                    fprintf('Processing window %d\n',j)
                end
                thisInd = self.window_ind(:,j);
                self.DMD_one_window(j, thisInd);
                self.get_library_modes(j);
            end
            
        end
        
        function set_default_windows(self)
            %Set some defaults for the window sizes
            %   Needs to be done after the data is imported
            if self.window_size==0
                self.window_size = floor(self.sz(2)/10);
            end
            if self.window_step==0
                self.window_step = floor(self.window_size/7);
            end
        end
        
        function set_window_indices(self)
            %Returns the indices of the windows
            numW = floor(...
                (self.sz(2) - self.window_size) / self.window_step) + 1;
            window_i = zeros(self.window_size, numW);
            
            for j=1:numW
                iStart = (j-1)*self.window_step + 1;
                iEnd = iStart + self.window_size - 1;
                window_i(:,j) = (iStart:iEnd)';
            end
            
            self.num_clusters = numW;
            self.window_ind = window_i;
        end
        
        function DMD_one_window(self, index, dat_indices)
            %Does DMD on a single file and saves the results
            vec = index; %Initial layer is 1; only a single time bin
            key = self.vec2key(vec);
            this_dat = self.dat(:,dat_indices);
            this_settings = self.plotter_set;
            this_settings.tspan = dat_indices*self.dt;
            
            self.PlotterDmd_all(key) = ...
                PlotterDmd(this_dat, this_settings);
        end
        
        function get_library_modes(self, j)
            %Determines which modes of the given window are 'interesting'
            %   Here 'interesting' means that the energy is >1sigma above
            %   the average value by default
            
            key = self.vec2key(j);
            thisDat = self.PlotterDmd_all(key);
            %             thisEnergy = abs(thisDat.coeff_sort).^2;
            %Want to do an expectation value, not just the basic energy
            tspan = (1:size(thisDat.Phi,2)) * self.dt;
            thisEnergy = trapz(abs(...
                thisDat.coeff_sort .* exp(thisDat.omega_sort.*tspan)).^2);
            thisFreq = thisDat.omega_sort;
            %             thisMean = mean(thisEnergy);
            thisMedian = median(thisEnergy);
            thisStd = std(thisEnergy);
            
            indices = self.sort_func(...
                thisEnergy, thisFreq, thisMedian, thisStd);
            thisLib = thisDat.phi_sort(:,indices);
            if self.lib_use_coeff_sign
                %Include the coefficient sign to increase comparibility
                thisSigns = repmat(sign(real(thisDat.coeff_sort))',...
                    size(thisLib,1),1);
                thisLib = thisLib.*thisSigns(:,indices);
            elseif self.lib_use_abs
                thisLib = abs(thisLib);
            end
            
            if isempty(find(isnan(thisLib), 1))
                self.dat_library = [self.dat_library thisLib];
                if size(indices,2)~=1
                    indices = indices';
                end
                save_ind = [ones(size(indices))*j, indices];
                self.dat_lib_ind = [self.dat_lib_ind; save_ind];
            else
                if self.verbose
                    warning('Nan values found; not including those modes in the library')
                end
            end
        end
        
        function copy_from_MultiResDmd(self, file_or_dat)
            %Copies settings etc from MultiResDmd object
            names = fieldnames(file_or_dat);
            for j=1:length(names)
                thisN = names{j};
                if isprop(self,thisN)
                    if ~strcmp(thisN,'dat')
                        self.(thisN) = file_or_dat.(thisN);
                    else
                        self.dat = file_or_dat.dat_layers(self.dat_key);
                    end
                end
            end
        end
        
        function get_reconstruction(self, num_modes, which_clusters)
            %Gets reconstruction from (optionally) some clusters
            if ~exist('which_clusters','var')
                which_clusters = [];
            end
            
            dat_approx = zeros(self.sz);
            for jW=1:self.num_clusters
                %Size of this window
                wSz = [self.sz(1), self.window_size];
                thisInd = self.window_ind(:,jW);
                
                tspan = thisInd*self.dt;
                if self.t0_each_bin
                    tspan = tspan - tspan(1);
                end
                
                key = self.vec2key(jW);
                %The PlotterDmd object has a method to directly reconstruct
                thisDMD = self.PlotterDmd_all(key);
                if isempty(which_clusters)
                    %Then get all modes
                    thisW = thisDMD.get_reconstruction(num_modes,tspan);
                else
                    %Then get a subset of modes, simply summing their
                    %contributions
                    thisW = zeros(wSz);
                    for jC=which_clusters
                        thisWindInd = self.get_modes_from_cluster(jW, jC);
                        if ~isempty(thisWindInd)
                            thisW = thisW + ...
                                thisDMD.get_reconstruction(thisWindInd,tspan);
                        end
                    end
                end
                
                dat_approx(:,thisInd) = dat_approx(:,thisInd) + ...
                    thisW./self.window_normalization(thisInd);
            end
            dat_approx = real(dat_approx);
            
            key = self.vec2key(which_clusters);
            self.approx_all(key) = dat_approx;
            
        end
        
        function thisModeWindInd =...
                get_modes_from_cluster(self, window_num, cluster_num)
            %Gets the modes corresponding to a single window and cluster
            %identity
            %   Note: cluster_num=0 means all non-identified clusters
            if isempty(self.dat_idx)
                error('No clustering data')
            end
            
            if cluster_num ~= 0
                %Get indices in the library then in the window
                thisCat = self.dat_idx==cluster_num;
                thisModeLibInd = self.dat_lib_ind(:,1)==window_num;
                thisModeWindInd = self.dat_lib_ind(...
                    logical(thisModeLibInd.*thisCat), 2);
            else
                %Call recursively to get all non-cluster modes
                clusterInd = self.get_modes_from_cluster(window_num, 1);
                for j=2:self.num_clusters
                    clusterInd = [clusterInd; ...
                        self.get_modes_from_cluster(window_num, j)]; %#ok<AGROW>
                end
                %The dmdplotter might have fewer modes than the full size
                %because it threw out some low-energy modes
                key = self.vec2key(window_num);
                thisDMD = self.PlotterDmd_all(key);
                thisModeWindInd = ones(size(thisDMD.coeff_sort));
                thisModeWindInd(clusterInd) = 0;
                thisModeWindInd = find(thisModeWindInd);
            end
        end
        
        function set_window_normalization(self)
            %Sets a vector with entries equal to the number of time a
            %particular time index is part of a window (for use in
            %normalization of the sums of multiple windows)
            
            wNorm = zeros(1,self.sz(2));
            for j=1:self.num_clusters
                thisInd = self.window_ind(:, j);
                wNorm = wNorm + ismember(1:self.sz(2), thisInd);
            end
            self.window_normalization = wNorm;
        end
        
        function plot_window_callback(self,...
                ~, event_data, use_coeff_sign, use_real_omega, subplot_func)
            %Plots a new power spectrum
            if ~exist('subplot_func','var')
                subplot_func = @plot_power_and_data;
            end
            x = event_data.IntersectionPoint(1);
            bar_x = linspace(1,self.sz(2),self.num_clusters);
            index = find( abs(x-bar_x)<1e-2 );
            delete(subplot(2,1,2))
            self.plot_power_and_data(index, false,...
                use_coeff_sign, use_real_omega, subplot_func); %#ok<FNDSB>
        end
        
    end
    
end

