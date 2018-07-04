classdef PlotterDmd < handle & SettingsImportableFromStruct
    %PlotterDmd: imports data, performs DMD, and visualizes
    %   This class is just a bunch of plotter options built in
    %
    % INPUTS
    %   file_or_dat - filename or data matrix; could also be MultiResDmd object
    %   settings - Struct of settings
    %
    % OUTPUTS -
    %   PlotterDmd object - object that does DMD on data and has many
    %   plotting options
    %
    %
    % Dependencies
    %   .m files, .mat files, and MATLAB products required:(updated on 06-Dec-2017)
    %             MATLAB (version 9.2)
    %             Curve Fitting Toolbox (version 3.5.5)
    %             System Identification Toolbox (version 9.6)
    %             Optimization Toolbox (version 7.6)
    %             Simulink Control Design (version 4.5)
    %             Statistics and Machine Learning Toolbox (version 11.1)
    %             Computer Vision System Toolbox (version 7.3)
    %             DMD.m
    %             prox_func.m
    %             settings_importable_from_struct.m
    %             xgeqp3_m.mexw64
    %             xormqr_m.mexw64
    %             optdmd.m
    %             varpro2.m
    %             varpro2dexpfun.m
    %             varpro2expfun.m
    %             varpro_lsqlinopts.m
    %             varpro_opts.m
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
    
    properties (SetAccess=private, Hidden=true)
        %Set in initializer
        dat
        raw
        %Raw DMD algorithm outputs
        A % matrix of dynamics
        coeff
        Omega
        Phi
        
        I_sort
        coeff_signs
        sz
    end
    
    properties (SetAccess=private, Hidden=false)
        %Sorted versions and metadata
        coeff_sort
        omega_sort
        phi_sort
    end
    
    properties (SetAccess={?SettingsImportableFromStruct})
        %Set in initializer
        verbose         = true
        filename char
        %Options
        dt              = 1
        tspan           = []
        dmd_percent      = 1
        to_subtract_mean  = false
        sort_mode        = ''
        model_order      = -1
        use_scaling      = false
        use_fb_dmd        = false
        use_optdmd       = true
        
        to_truncate_omega = false
        which_truncate_omega = false
    end
    
    methods
        function self = PlotterDmd(file_or_dat, settings)
            %% Import user settings
            self.import_settings_to_self(settings);
            %==========================================================================
            
            %% Import data and preprocess
            if ischar(file_or_dat)
                self.filename = file_or_dat;
                self.raw = importdata(file_or_dat);
            elseif isnumeric(file_or_dat)
                %Assume the user passed the data directly
                self.filename = '';
                self.raw = file_or_dat;
            end
            
            if self.model_order==0
                self.model_order = size(self.raw,1);
            end
            self.preprocess();
            
            self.sz = size(self.dat);
            self.which_truncate_omega = [];
            %==========================================================================
            %% Do dmd
            self.do_dmd();
            %==========================================================================
            %% Process data
            % This is mostly sorting and cutting off large real parts, if
            % those settings are checked
            self.postprocess();
            %==========================================================================
            
        end
        
        function Xapprox = get_reconstruction(self, ind, tspan)
            % Reconstructs data using the DMD modes specified
            %   Can specify scalar or vector of modes
            max_ind = length(self.Omega);
            if ~exist('ind','var')
                ind = 0;
            end
            if ~exist('tspan','var')
                tspan = self.dt * (1:size(self.raw,2));
            end
            if isscalar(ind)
                if ind>max_ind
                    ind = 1:min(ind,max_ind);
                elseif ind<1
                    ind = 1:max_ind;
                else
                    disp('Plotting a single mode')
                end
            else
                assert(max(ind)<=max_ind,...
                    'Mode index out of range')
                %                 ind = 1:min(ind(end),max_ind);
            end
            
            Omega_approx = self.omega_sort(ind);
            coeff_approx = self.coeff_sort(ind);
            
            for jT = length(tspan):-1:1
                Xmodes(:,jT) = coeff_approx.*exp(Omega_approx*tspan(jT));
            end
            Xapprox = self.phi_sort(:,ind)*Xmodes;
            if self.to_subtract_mean
                for jM=1:size(Xapprox,1)
                    Xapprox(jM,:) = Xapprox(jM,:) + mean(self.raw(jM,:));
                end
            end
        end
        
        function error = calc_reconstruction_error(self, indices)
            % Uses L2 norm to get full error
            if ~exist('indices','var') || isempty(indices)
                indices = 1:size(self.dat,1);
            end
            dat_approx = self.get_reconstruction();
            error = norm(self.dat(indices,:) - dat_approx(indices,:));
        end
        
        function Xapprox = get_reconstructed_endpoint(self,...
                ind, t_end, x0)
            %Reconstructs data using the DMD modes specified
            %   Can specify scalar or vector of modes
            max_ind = length(self.Omega);
            if isscalar(ind)
                if ind>max_ind
                    ind = 1:min(ind,max_ind);
                else
                    disp('Plotting a single mode')
                end
            elseif ischar(ind) || isempty(ind)
                %Defaults to all modes
                ind = 1:max_ind;
            else
                assert(max(ind)<=max_ind,'Mode index out of range')
            end
            
            Omega_approx = self.omega_sort(ind);
            Phi_approx = self.phi_sort(:,ind);
            if length(x0)==length(ind)
                % Assume we have coefficients in DMD mode basis
                this_coeff = x0;
            elseif length(x0)==size(Phi_approx,1)
                % Then x0 is in the data basis and needs to be in DMD mode
                % basis
                this_coeff = Phi_approx\x0;
            else
                error("Unrecognized initial point size")
            end
            
            Xapprox = Phi_approx*(this_coeff.*exp(Omega_approx*t_end));
            
            if self.to_subtract_mean
                for jM=1:size(Xapprox,1)
                    Xapprox(jM,:) = Xapprox(jM,:) + mean(self.raw(jM,:));
                end
            end
        end
    end
    
    methods % Plotting
        function plot_coeff(self)
            figure;
            plot(abs(self.coeff_sort),'o')
            title('sorted coefficients')
        end
        
        function plot_modes(self, numModes)
            %Works with a scalar or vector numModes input
            if isscalar(numModes)
                useModes = 1:numModes;
            elseif isvector(numModes)
                useModes = numModes;
                numModes = length(numModes);
            end
            
            figure;
            %Rescale the modes by the sign of their coefficient
            plot(real(self.phi_sort(:,useModes)) .* ...
                self.coeff_signs(:,useModes),...
                'LineWidth',2)
            legend(self.getLegendStr(numModes))
            title(sprintf('First %d DMD modes',numModes))
        end
        
        function plot_modes_prox(self, numModes, lambda)
            %Works with a scalar or vector numModes input
            if isscalar(numModes)
                useModes = 1:numModes;
            elseif isvector(numModes)
                useModes = numModes;
                numModes = length(numModes);
            end
            figure;
            hold on;
            for j=1:numModes
                plot(self.coeff_signs(useModes(j))*...
                    prox_func(real(self.phi_sort(:,useModes(j))),lambda),...
                    'LineWidth',2)
            end
            title(sprintf('First %d DMD modes',numModes))
            legend(self.getLegendStr(numModes))
        end
        
        function plot_3modes(self)
            figure;
            scatter3(self.phi_sort(:,1),self.phi_sort(:,2),self.phi_sort(:,3),...
                log(abs(1e10./real(self.omega_sort))),imag(self.omega_sort), 'filled')
            
            colorbar
            title('Neurons in 3 DMD modes: color is imag(frequency); size is 1/real(omega)')
            xlabel('Mode 1')
            ylabel('Mode 2')
            zlabel('Mode 3')
        end
        
        function plot_2modes_neurNum(self)
            figure;
            scatter3(...
                self.phi_sort(:,1),self.phi_sort(:,2),1:size(self.Phi,2),...
                log(abs(1e10./real(self.omega_sort))),imag(self.omega_sort),...
                'filled')
            
            colorbar
            title('Neurons in 3 DMD modes: color is imag(frequency); size is 1/real(omega)')
            xlabel('Mode 1')
            ylabel('Mode 2')
            zlabel('Neuron number')
        end
        
        function plot_omega_modeNum(self, numModes, sizes)
            %Plots frequency with mode number as a third dimension
            if ~exist('numModes','var')
                modeInd = 1:length(self.Omega);
            else
                modeInd = 1:numModes;
            end
            if ~exist('sizes','var')
                sizes = abs(self.coeff_sort(modeInd));
            end
            
            om = self.omega_sort(modeInd);
            scatter3(real(om), imag(om), modeInd, sizes)
            xlabel('Real part')
            ylabel('Imaginary part')
            zlabel('Mode index')
            
        end
        
        function plot_power_spectrum(self,...
                useCoeffSign, useRealOmega, categories)
            %Plots an analog of an fft power spectrum
            if ~exist('useRealOmega','var')
                useRealOmega = false;
            end
            if ~exist('useCoeffSign','var')
                useCoeffSign = false;
            end
            
            
            freq = abs(imag(self.omega_sort)) / (2*pi);
            
            if self.use_scaling
                power = zeros(size(freq));
                for j=1:length(freq)
                    power(j) = sum(abs(self.phi_sort(:,j)).^2);
                end
            else
                %If not scaled, just use the coefficients directly
                power = log(abs(self.coeff_sort).^2);
            end
            if useRealOmega
                power = power.*abs(1/real(self.omega_sort));
            end
            if useCoeffSign
                power = power.*sign(real(self.coeff_sort));
            end
            
            %figure;
            if ~exist('categories','var')
                plot(freq, power, 'o', 'LineWidth', 2)
            else
                hold on
                %Set the proper color by hand
                plot_colors = lines(max(categories));
                for jC = unique(categories)'
                    ind = find(categories==jC);
                    if jC==0
                        plot(freq(ind), power(ind), 'ok')
                    else
                        plot(freq(ind), power(ind), 'o', 'LineWidth', 2)
                        f = gca();
                        f.Children(1).Color = plot_colors(jC,:);
                    end
                end
                hold off
            end
            ylabel('log(Power of mode)')
            xlabel('Frequency (Hz)')
        end
        
        function plot_eigenvalues(self, ~, ~, categories)
            %Plots the dmd eigenvalues
            
            freq = self.omega_sort;
            
            %figure;
            if ~exist('categories','var')
                plot(freq, 'o', 'LineWidth', 2)
            else
                hold on
                %Set the proper color by hand
                plot_colors = lines(max(categories));
                for jC = unique(categories)'
                    ind = find(categories==jC);
                    if jC==0
                        plot(freq(ind), power(ind), 'ok')
                    else
                        plot(freq(ind), power(ind), 'o', 'LineWidth', 2)
                        f = gca();
                        f.Children(1).Color = plot_colors(jC,:);
                    end
                end
                hold off
            end
            ylabel('Eigenvalue (imag)')
            xlabel(sprintf('Eigenvalue (real); max=%.2f',...
                max(real(freq))))
        end
    end
    
    methods (Access = private)
        function legend_str = getLegendStr(~,numModes)
            legend_str = cell(numModes,1);
            for j2=1:numModes
                legend_str{j2} = sprintf('Mode %d',j2);
            end
        end
        
        function preprocess(self)
            sz_raw = size(self.raw);
            if self.to_subtract_mean
                for jM=1:sz_raw(1)
                    self.dat(jM,:) = self.raw(jM,:) - mean(self.raw(jM,:));
                end
            else
                self.dat = self.raw;
            end
            
            if self.model_order>0
                self.model_order = min([self.model_order,sz_raw]);
            else
                self.model_order = min(sz_raw);
            end
        end
        
        function postprocess(self)
            % First, sorting
            
            switch self.sort_mode
                case ''
                    %No sorting by default
                    self.I_sort = 1:size(self.coeff);
                    self.coeff_sort = self.coeff;
                case 'abs'
                    [self.coeff_sort, self.I_sort] = ...
                        sort(abs(self.coeff));
                case 'real'
                    [self.coeff_sort, self.I_sort] = ...
                        sort(real(self.coeff));
                otherwise
                    error('Unrecognized sort mode')
            end
            
            self.omega_sort = self.Omega(self.I_sort);
            self.phi_sort = self.Phi(:,self.I_sort);
            
            self.coeff_signs = repmat(sign(real(self.coeff))', ...
                size(self.phi_sort, 1), 1);
            
            %Second, if the option is checked, throw out all positive real
            %parts of the frequencies... these will blow up in the
            %reconstruction and are generally problematic
            for j = 1:length(self.omega_sort)
                if real(self.omega_sort(j)) > 0
                    self.omega_sort(j) = 1i*imag(self.omega_sort(j));
                end
            end
            
        end
        
        function do_dmd(self)
            %Does either my own dmd routine, or Travis' optimized dmd
            
            if ~self.use_optdmd
                %Use my own DMD algorithm writtten in MATLAB
                [self.coeff, self.Omega, self.Phi, self.A ] = ...
                    dmd(self.dat, self.dt,...
                    self.model_order, self.dmd_percent, self.use_scaling,...
                    self.use_fb_dmd);
            else
                %Use Travis' optimized DMD written in mostly C
                self.tspan = 0:self.dt:self.sz(2);
                self.tspan = self.tspan(1:end-1);
%                 numModes = round(self.dmd_percent*self.sz(1)*0.5);%This is hacky
%                 numModes = min(numModes, self.sz(2)-1);
                fitFullData = 1;
                
                [self.Phi, self.Omega, self.coeff, self.A] = ...
                    optdmd(self.dat, self.tspan, self.model_order, fitFullData);
            end
        end
    end
    
end

