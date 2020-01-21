classdef AbstractWindowDmd < SettingsImportableFromStruct
    %Abstract super-class for WindowDmd
    %   Implements the basic architecture for setting up the windows, so
    %   that I can use it in other functions as well
    
    properties (SetAccess={?SettingsImportableFromStruct}, Hidden=true)
        % For windows
        window_size
        window_step
    end
    
    properties
        % General data
        sz
        % For windows
        num_clusters
        window_ind
        window_normalization
    end
    
    methods
        
        function self = AbstractWindowDmd(settings)
            
            % Set defaults and import settings
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_abstract_defaults();
            self.import_settings_to_self(settings);
        end
        
        function setup_windows(self, sz)
            % Just give the size of the data, then generate all the windows
            self.sz = sz;
            self.set_default_windows();
            self.set_window_indices();
            self.set_window_normalization();
            
        end
        
        function set_default_windows(self)
            %Set some defaults for the window sizes
            %   Needs to be done after the data is imported
            if isempty(self.window_size)
                self.window_size = floor(self.sz(2)/10);
            else
                assert(self.window_size > self.sz(1),...
                    'Window size must be greater than the number of channels')
            end
            if isempty(self.window_step)
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
        
        function set_abstract_defaults(self)
            defaults = struct(...
                'window_size',[],...
                'window_step',[]);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
        end
    end
end

