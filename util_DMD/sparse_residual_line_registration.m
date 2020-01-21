function [registered_lines, registered_names] = ...
    sparse_residual_line_registration(all_U, all_acf, model_base)
% Given a cell array of control signals and a model object with a get_names
% method, try to follow the control signals across the different fits using
% the neurons/channels that they input onto
%   i.e. use the matrix B in X2=A*X1+B*U as an "importance matrix" for
%   where the control signal U goes in the original data space (even if B
%   isn't sparsified)
X1 = model_base.dat(:,1:end-1);
X2 = model_base.dat(:,2:end);
n = model_base.dat_sz(1);

max_rank = length(all_U);
% Might start at a rank that isn't 1
if size(all_U{1},1) > 1
    rank_vec = size(all_U{1},1):(size(all_U{1},1)+max_rank-1);
else
    rank_vec = 1:max_rank;
end
all_names = cell(max_rank, 1);
registered_lines = {};
registered_names = {};
for i = 1:max_rank
    all_names{i} = cell(1, i);
    U = all_U{i};
    AB = X2 / [X1; U];
%     A = AB(:, 1:n);
    B = AB(:, (n+1):end);
    for i2 = 1:rank_vec(i)
        x = abs(B(:, i2));
        [this_maxk, this_ind] = maxk(x, 2);
        if this_maxk(1)/this_maxk(2) > 2
            this_ind = this_ind(1);
        end
        
        % Names for registered lines graphing
        all_names{i}{:,i2} = model_base.get_names(this_ind, true);
        if ischar(all_names{i}{:,i2})
            all_names{i}{:,i2} = {all_names{i}{:,i2}};
        end
        
        % Only feature: max acf
        signal_name = sort(all_names{i}{:,i2});
        found_registration = false;
        [this_y, which_sparsity_in_rank] = max(all_acf{i}(i2, :));
        
        which_rank = i;
        which_line_in_rank = i2;
        this_dat = table(which_rank, this_y, which_line_in_rank,...
            which_sparsity_in_rank);
        for i4 = 1:length(registered_names)
            % Attach them to a previous line if it exists
            if isequal(registered_names{i4}, signal_name)
                registered_lines{i4} = [registered_lines{i4}; ...
                    this_dat];  %#ok<AGROW>
                found_registration = true;
                break
            end
        end
        if ~found_registration
            if isempty(registered_names)
                registered_names = {signal_name};
            else
                registered_names = [registered_names {signal_name}]; %#ok<AGROW>
            end
            registered_lines = [registered_lines ...
                {this_dat} ]; %#ok<AGROW>
        end
    end
end

end

