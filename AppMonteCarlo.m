function AppMonteCarlo()
% APP_MONTECARLO - Professional Network Analysis Dashboard (v2.0)
% =========================================================================
% AUTHOR:      [Your Name]
% DATE:        2024
% DESCRIPTION: 
%   Updated to support Wang et al. "Effective Latency" metrics.
%   Includes:
%     - Doppler/Jitter Analysis
%     - Earth Texture Mapping
%     - Orbital Plane Visualization
% =========================================================================

    % --- 1. INITIALIZATION ---
    close all; clc;
    
    % Professional Color Palette (Dark Blue/Grey Theme)
    C.Bg       = [0.95 0.95 0.97]; 
    C.Panel    = [1.00 1.00 1.00]; 
    C.Primary  = [0.00 0.48 0.75]; % MATLAB Blue
    C.Success  = [0.20 0.65 0.35]; % Green
    C.Warning  = [0.85 0.55 0.10]; % Orange
    C.Danger   = [0.75 0.20 0.20]; % Red
    C.TextDark = [0.15 0.15 0.20]; 
    C.TextDim  = [0.50 0.50 0.55];

    % Data Storage
    State.LastPos = [];
    State.LastVel = [];
    State.LastPathBase = [];
    State.LastPathWang = [];
    State.Metrics = struct(); % To hold specific route details
    State.HasRun = false;
    
    % Load Earth Topography (Built-in MATLAB data)
    try
        earthData = load('topo.mat');
        State.Topo = earthData.topo;
        State.TopoLegend = earthData.topomap1;
    catch
        State.Topo = []; % Fallback to plain sphere
    end
    
    % Main Window
    fig = uifigure('Name', 'LEO Network Reliability Analyzer (Wang et al.)', ...
        'Position', [50 50 1400 850], 'Color', C.Bg);
        
    % Main Layout
    gridMain = uigridlayout(fig, [1, 2]); 
    gridMain.ColumnWidth = {360, '1x'}; 
    gridMain.Padding = [15 15 15 15];
    gridMain.ColumnSpacing = 15;
    
    % =====================================================================
    % 2. SIDEBAR (CONTROLS & RESULTS)
    % =====================================================================
    pSide = uipanel(gridMain, 'BackgroundColor', C.Bg, 'BorderType', 'none');
    gSide = uigridlayout(pSide, [5, 1]); 
    
    gSide.RowHeight = {160, 130, 100, 200, '1x'}; 
    gSide.Padding = [0 0 0 0]; gSide.RowSpacing = 15;
    
    % --- CARD 1: CONFIGURATION ---
    card1 = createCard(gSide, 'SIMULATION SETUP');
    lc1 = uigridlayout(card1, [4, 2]); 
    lc1.Padding = [15 10 15 10]; 
    lc1.RowHeight = {25, 25, 25, 25};
    lc1.RowSpacing = 8;
    
    uilabel(lc1, 'Text', 'Constellation Size (N):', 'FontColor', C.TextDark);
    efN = uieditfield(lc1, 'numeric', 'Value', 1000, 'Limits', [100 5000]);
    
    uilabel(lc1, 'Text', 'Monte Carlo Trials:', 'FontColor', C.TextDark);
    efTrials = uieditfield(lc1, 'numeric', 'Value', 100, 'Limits', [10 5000]);
    
    % --- CARD 2: ALGORITHM TUNING ---
    card2 = createCard(gSide, 'ROUTING STRATEGY');
    lc2 = uigridlayout(card2, [3, 1]); 
    lc2.Padding = [15 10 15 10]; 
    lc2.RowHeight = {20, 40, 20};
    
    uilabel(lc2, 'Text', 'Reliability Weight (\alpha)', 'FontWeight', 'bold', 'FontColor', C.Primary);
    sldAlpha = uislider(lc2, 'Limits', [0 100], 'Value', 80); % Default higher for Wang
    lblAlpha = uilabel(lc2, 'Text', 'Balance: 80% Effective Latency / 20% Distance', ...
        'HorizontalAlignment', 'center', 'FontColor', C.TextDim, 'FontSize', 10);
    
    sldAlpha.ValueChangedFcn = @(s,e) updateLabel(lblAlpha, s.Value);

    % --- CARD 3: EXECUTION ---
    card3 = createCard(gSide, 'CONTROL');
    lc3 = uigridlayout(card3, [1, 2]); 
    lc3.Padding = [20 20 20 20]; 
    lc3.ColumnWidth = {'1x', 50};
    
    btnRun = uibutton(lc3, 'Text', 'RUN ANALYSIS', ...
        'BackgroundColor', C.Primary, 'FontColor', 'w', 'FontWeight', 'bold', 'FontSize', 14);
    
    lampStatus = uilamp(lc3, 'Color', [0.8 0.8 0.8]); 
    
    % --- CARD 4: RESULTS DASHBOARD ---
    pRes = uipanel(gSide, 'BackgroundColor', 'w', 'Title', 'AGGREGATE METRICS', ...
        'FontWeight', 'bold', 'FontSize', 11, 'ForegroundColor', C.TextDim);
    
    lRes = uigridlayout(pRes, [6, 2]); 
    lRes.RowHeight = {15, 30, 15, 30, 15, 30}; 
    lRes.Padding=[15 10 15 10];
    
    % Labels
    uilabel(lRes, 'Text', 'Avg Latency Penalty', 'FontColor', C.TextDim, 'FontSize', 10);
    uilabel(lRes, 'Text', 'Avg SNR Gain', 'FontColor', C.TextDim, 'FontSize', 10);
    
    % Values
    res_Lat = uilabel(lRes, 'Text', '-- ms', 'FontWeight','bold', 'FontSize',18, 'FontColor', C.Warning);
    res_Gain = uilabel(lRes, 'Text', '-- dB', 'FontWeight','bold', 'FontSize',18, 'FontColor', C.Success);
    
    % Doppler Labels
    uilabel(lRes, 'Text', 'Max Doppler Shift', 'FontColor', C.TextDim, 'FontSize', 10);
    uilabel(lRes, 'Text', 'Jitter (Doppler Std)', 'FontColor', C.TextDim, 'FontSize', 10);
    
    % Doppler Values
    res_Dop = uilabel(lRes, 'Text', '-- kHz', 'FontWeight','bold', 'FontSize',18, 'FontColor', C.TextDark);
    res_Jit = uilabel(lRes, 'Text', '-- Hz', 'FontWeight','bold', 'FontSize',18, 'FontColor', C.TextDark);

    % 3D Button
    btn3D = uibutton(gSide, 'Text', 'Open 3D Inspector 🌍', ...
        'BackgroundColor', [0.2 0.2 0.2], 'FontColor', 'w', 'Enable', 'off', 'FontSize', 12); 

    % =====================================================================
    % 3. VISUALIZATION AREA (TABS)
    % =====================================================================
    tabGroup = uitabgroup(gridMain);
    
    % Tab 1: Latency Distribution
    t1 = uitab(tabGroup, 'Title', 'Latency & Cost');
    gT1 = uigridlayout(t1, [1, 1]);
    axLat = uiaxes(gT1); title(axLat, 'End-to-End Latency Distribution'); grid(axLat,'on');
    
    % Tab 2: Reliability
    t2 = uitab(tabGroup, 'Title', 'Reliability (SNR)');
    gT2 = uigridlayout(t2, [1, 1]);
    axRel = uiaxes(gT2); title(axRel, 'Link Budget Improvement'); grid(axRel,'on');
    
    % Tab 3: Doppler/Jitter (NEW)
    t3 = uitab(tabGroup, 'Title', 'Doppler Dynamics');
    gT3 = uigridlayout(t3, [2, 1]);
    axDop = uiaxes(gT3); title(axDop, 'Max Doppler Shift per Route (kHz)'); grid(axDop,'on');
    axJit = uiaxes(gT3); title(axJit, 'Route Jitter (Doppler Std Dev)'); grid(axJit,'on');
    
    % =====================================================================
    % 4. CALLBACKS
    % =====================================================================
    
    btnRun.ButtonPushedFcn = @(b,e) runSimulation();
    btn3D.ButtonPushedFcn = @(b,e) show3DView();
    
    function runSimulation()
        btnRun.Enable = 'off'; lampStatus.Color = 'y'; drawnow;
        
        try
            % 1. Inputs
            N = efN.Value;
            trials = efTrials.Value;
            P.Range = 3500; 
            P.UseOpt = true; 
            P.Wang_Alpha = sldAlpha.Value; 
            
            % 2. Generate Universe
            [pos, vel, ~] = SimUtils.generateConstellation(N);
            [G_Base, G_Wang] = SimUtils.buildGraphs(pos, vel, [], P);
            
            % 3. Monte Carlo Loop
            lat_b = []; lat_w = []; 
            gain = [];
            dop_b = []; dop_w = [];
            jit_b = []; jit_w = [];
            
            bins = conncomp(G_Wang);
            exampleFound = false;
            
            % Use Waitbar for UX
            wb = waitbar(0, 'Running Monte Carlo...');
            
            for k = 1:trials
                nodes = randperm(N, 2);
                u = nodes(1); v = nodes(2);
                
                if bins(u) == bins(v)
                    path_b = shortestpath(G_Base, u, v);
                    path_w = shortestpath(G_Wang, u, v);
                    
                    if ~isempty(path_b) && ~isempty(path_w)
                        % Get Detailed Metrics (Now returns 4 values)
                        [Lb, Jb, Fb, Db] = SimUtils.getPathMetrics(path_b, pos, vel, [], []);
                        [Lw, Jw, Fw, Dw] = SimUtils.getPathMetrics(path_w, pos, vel, [], []);
                        
                        lat_b(end+1) = Lb;  lat_w(end+1) = Lw;
                        gain(end+1) = Fb - Fw; 
                        dop_b(end+1) = Db;  dop_w(end+1) = Dw;
                        jit_b(end+1) = Jb;  jit_w(end+1) = Jw;
                        
                        % Save interesting case (Different paths)
                        if ~exampleFound && length(path_b) ~= length(path_w)
                            State.LastPos = pos;
                            State.LastVel = vel;
                            State.LastPathBase = path_b;
                            State.LastPathWang = path_w;
                            
                            % Store specific metrics for the popup
                            State.Metrics.Std = [Lb, Fb, Db, Jb];
                            State.Metrics.Wang = [Lw, Fw, Dw, Jw];
                            
                            exampleFound = true;
                        end
                    end
                end
                if mod(k,10)==0, waitbar(k/trials, wb); end
            end
            close(wb);
            
            % 4. Update UI
            updatePlots(lat_b, lat_w, gain, dop_b, dop_w, jit_b, jit_w);
            updateStats(lat_b, lat_w, gain, dop_w, jit_w);
            
            if exampleFound
                State.HasRun = true;
                btn3D.Enable = 'on';
            end
            lampStatus.Color = 'g';
            
        catch ME
            if exist('wb','var'), close(wb); end
            uialert(fig, ME.message, 'Error');
            lampStatus.Color = 'r';
        end
        btnRun.Enable = 'on';
    end

    function updatePlots(lb, lw, g, db, dw, jb, jw)
        % Tab 1: Latency
        cla(axLat); hold(axLat, 'on');
        histogram(axLat, lb, 30, 'FaceColor', 'k', 'FaceAlpha', 0.3);
        histogram(axLat, lw, 30, 'FaceColor', C.Primary, 'FaceAlpha', 0.6);
        legend(axLat, 'Shortest Path', 'Effective Latency (Wang)');
        xlabel(axLat, 'Latency (ms)'); ylabel(axLat, 'Count');
        
        % Tab 2: Gain
        cla(axRel); 
        histogram(axRel, g, 30, 'FaceColor', C.Success);
        xlabel(axRel, 'SNR Improvement (dB)'); title(axRel, 'Reliability Gain');
        xline(axRel, mean(g), '--r', 'LineWidth', 2);
        
        % Tab 3: Doppler
        cla(axDop); hold(axDop, 'on');
        histogram(axDop, db/1000, 20, 'FaceColor', 'k', 'FaceAlpha', 0.3);
        histogram(axDop, dw/1000, 20, 'FaceColor', C.Warning, 'FaceAlpha', 0.6);
        xlabel(axDop, 'Max Doppler (kHz)'); legend(axDop, 'Std', 'Reliable');
        
        cla(axJit); hold(axJit, 'on');
        histogram(axJit, jb, 20, 'FaceColor', 'k', 'FaceAlpha', 0.3);
        histogram(axJit, jw, 20, 'FaceColor', C.Danger, 'FaceAlpha', 0.6);
        xlabel(axJit, 'Doppler Jitter (Hz)');
    end

    function updateStats(lb, lw, g, dw, jw)
        res_Lat.Text = sprintf('+%.1f ms', mean(lw - lb));
        res_Gain.Text = sprintf('+%.1f dB', mean(g));
        res_Dop.Text = sprintf('%.1f kHz', mean(dw)/1000);
        res_Jit.Text = sprintf('%.1f Hz', mean(jw));
    end

    function show3DView()
        if ~State.HasRun, return; end
        
        f3 = figure('Name', '3D Constellation Inspector', 'Color', 'k', 'NumberTitle', 'off');
        ax3 = axes(f3, 'Color', 'k'); hold(ax3, 'on'); axis(ax3, 'equal');
        
        % --- 1. Draw Earth (Textured) ---
        R = 6378;
        if ~isempty(State.Topo)
            [x,y,z] = sphere(50);
            props.AmbientStrength = 0.1;
            props.DiffuseStrength = 1;
            props.SpecularColorReflectance = .5;
            props.SpecularExponent = 20;
            props.FaceColor= 'texture';
            props.EdgeColor = 'none';
            props.FaceLighting = 'phong';
            props.CData = State.Topo; 
            surface(x*R, y*R, z*R, props, 'Parent', ax3);
        else
            [x,y,z] = sphere(50);
            surf(ax3, x*R, y*R, z*R, 'FaceColor', [0.1 0.1 0.15], 'EdgeColor', 'none', 'FaceAlpha', 0.9);
        end
        
        % --- 2. Draw Orbital Planes (Visual Guide) ---
        % Re-calculate plane geometry roughly based on node distribution
        pos = State.LastPos;
        plot3(ax3, pos(:,1), pos(:,2), pos(:,3), '.', 'Color', [0.4 0.4 0.4], 'MarkerSize', 4);
        
        % --- 3. Highlight Paths ---
        pb = State.LastPathBase;
        pw = State.LastPathWang;
        
        % Standard Path (Red)
        plot3(ax3, pos(pb,1), pos(pb,2), pos(pb,3), '-o', ...
            'Color', [1 0.3 0.3], 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'Standard');
        
        % Wang Path (Cyan)
        plot3(ax3, pos(pw,1), pos(pw,2), pos(pw,3), '--s', ...
            'Color', [0 1 1], 'LineWidth', 2, 'MarkerFaceColor', 'c', 'DisplayName', 'Wang (Reliable)');
        
        % --- 4. Route Inspector Text ---
        % Create a text box on the plot with the specific metrics
        mS = State.Metrics.Std; % [Lat, SNR, Dop, Jit]
        mW = State.Metrics.Wang;
        
        msg = {
            '\bf\color{white}SELECTED ROUTE METRICS';
            '---------------------------------';
            sprintf('\\color[rgb]{1,0.3,0.3}Standard (Red):');
            sprintf('  Lat: %.1f ms | SNR: %.1f dB', mS(1), mS(2));
            sprintf('  Dop: %.1f kHz | Jit: %.1f Hz', mS(3)/1000, mS(4));
            ' ';
            sprintf('\\color{cyan}Reliable (Cyan):');
            sprintf('  Lat: %.1f ms | SNR: %.1f dB', mW(1), mW(2));
            sprintf('  Dop: %.1f kHz | Jit: %.1f Hz', mW(3)/1000, mW(4));
        };
        
        text(ax3, 1.1*R, 1.1*R, 1.1*R, msg, 'BackgroundColor', [0 0 0 0.7], ...
            'EdgeColor', 'w', 'Margin', 5, 'VerticalAlignment', 'top');
        
        % Aesthetics
        view(ax3, 3); grid(ax3, 'on'); 
        legend(ax3, 'Location', 'southwest', 'TextColor', 'w', 'Color', 'none');
        title(ax3, '3D Route Inspection', 'Color', 'w');
        
        % Lighting
        light('Position', [1 0 0], 'Style', 'infinite');
        rotate3d(ax3, 'on');
    end

    % --- UI HELPER FUNCTIONS ---
    function p = createCard(parent, titleText)
        p = uipanel(parent, 'BackgroundColor', 'w', ...
            'Title', titleText, 'FontWeight', 'bold', 'FontSize', 10, ...
            'ForegroundColor', [0.6 0.6 0.6]);
    end

    function updateLabel(lbl, val)
        lbl.Text = sprintf('Balance: %d%% Effective Latency / %d%% Distance', round(val), 100-round(val));
    end
end
