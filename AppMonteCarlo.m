function RunMonteCarlo_Sensitivity()
% RUNMONTECARLO_SENSITIVITY - Compare Altitude Impacts (600km vs 1200km)
% -------------------------------------------------------------------------

    % --- CONFIGURATION ---
    clc; close all;
    Ns_List = [500, 1000, 1500, 2000, 3000]; 
    ITERATIONS = 50; 
    MAX_HOPS = 30;
    
    % Define Scenarios: [Altitude (m)]
    Scenarios = [600e3, 1200e3]; 
    ScenarioLabels = {'600 km', '1200 km'};
    Colors = {'#D95319', '#0072BD'}; % Orange, Blue
    
    % Storage: Cell array to hold results for each scenario
    Data_Lat = cell(1, 2);
    Data_Rel = cell(1, 2);
    Data_Hop = cell(1, 2);

    fprintf('Starting Sensitivity Analysis...\n');

    % --- OUTER LOOP: SCENARIOS ---
    for s_idx = 1:length(Scenarios)
        current_alt = Scenarios(s_idx);
        fprintf('\n--- Running Scenario: Altitude = %.0f km ---\n', current_alt/1000);
        
        % Temporary storage for this scenario
        res_lat = zeros(length(Ns_List), 1);
        res_rel = zeros(length(Ns_List), 1);
        res_hop = zeros(length(Ns_List), 1);
        
        % --- INNER LOOP: DENSITY ---
        for i = 1:length(Ns_List)
            Ns = Ns_List(i);
            
            lat_acc = []; rel_acc = []; hop_acc = [];
            
            % Parallel Monte Carlo
            parfor k = 1:ITERATIONS
                % 1. Generate Network with CUSTOM ALTITUDE
                [V, src, dst] = SimUtils.generateNetwork(Ns, current_alt);
                
                % 2. Find Route
                [route, success] = SimUtils.findRoute(V, src, dst, MAX_HOPS);
                
                % 3. Evaluate
                if success
                    m = SimUtils.evaluateRoute(V, route);
                    lat_acc = [lat_acc; m.TotalLatency]; %#ok<AGROW>
                    rel_acc = [rel_acc; m.Reliability]; %#ok<AGROW>
                    hop_acc = [hop_acc; m.Hops]; %#ok<AGROW>
                end
            end
            
            % Averages
            if ~isempty(lat_acc)
                res_lat(i) = mean(lat_acc) * 1000;
                res_rel(i) = mean(rel_acc);
                res_hop(i) = mean(hop_acc);
            end
            fprintf('Ns: %4d | Lat: %.2f ms | Rel: %.4f\n', Ns, res_lat(i), res_rel(i));
        end
        
        % Store scenario data
        Data_Lat{s_idx} = res_lat;
        Data_Rel{s_idx} = res_rel;
        Data_Hop{s_idx} = res_hop;
    end
    
    % --- PLOTTING COMPARISON ---
    plotComparison(Ns_List, Data_Lat, Data_Rel, Data_Hop, ScenarioLabels, Colors);
end

function plotComparison(x, lat_data, rel_data, hop_data, labels, colors)
    figure('Name', 'Altitude Sensitivity Analysis', 'Color', 'w', 'Position', [100 100 1200 500]);
    
    % 1. Latency
    subplot(1, 3, 1); hold on;
    for i = 1:2
        plot(x, lat_data{i}, '-o', 'LineWidth', 2, 'Color', colors{i}, ...
            'DisplayName', labels{i}, 'MarkerFaceColor', 'w');
    end
    grid on; legend('Location', 'best');
    xlabel('Number of Satellites'); ylabel('Effective Latency (ms)');
    title('Impact of Altitude on Latency');
    
    % 2. Reliability
    subplot(1, 3, 2); hold on;
    for i = 1:2
        plot(x, rel_data{i}, '-s', 'LineWidth', 2, 'Color', colors{i}, ...
            'DisplayName', labels{i}, 'MarkerFaceColor', 'w');
    end
    grid on; legend('Location', 'best');
    xlabel('Number of Satellites'); ylabel('E2E Reliability');
    title('Impact of Altitude on Reliability');
    ylim([0, 1.1]);
    
    % 3. Hops
    subplot(1, 3, 3); hold on;
    for i = 1:2
        plot(x, hop_data{i}, '-^', 'LineWidth', 2, 'Color', colors{i}, ...
            'DisplayName', labels{i}, 'MarkerFaceColor', 'w');
    end
    grid on; legend('Location', 'best');
    xlabel('Number of Satellites'); ylabel('Average Hop Count');
    title('Impact of Altitude on Routing Hops');
end