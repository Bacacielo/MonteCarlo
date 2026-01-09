classdef SimUtils
% SIMUTILS - High-Fidelity LEO Astrodynamics & Graph Theory Engine
    %
    % DESCRIPTION: 
    %   Core utility class for LEO satellite network simulation. 
    %   Implements orbital mechanics, topology construction, and routing metrics 
    %   based on the methodology of Wang et al. (2024).
    %
    % KEY FEATURES:
    %   1. Stochastic Topology: Binomial Point Process (BPP) on a Sphere.
    %   2. Routing Metric: Effective Latency considering ARQ retransmissions.
    %   3. Reliability: Physical layer modeling (Link Budget, Rayleigh Fading).
    %   4. Jitter Analysis: Doppler shift and packet delay variation.
    %
    % AUTHOR: chrisvasill
    % REPOSITORY: LEO-Network-Analyzer
    % =====================================================================
    
    properties (Constant)
        % --- EARTH PHYSICS (WGS-84) ---
        R_EARTH = 6378.137;           % Equatorial Radius (km)
        FLATTENING = 1/298.257223563; % Earth Flattening Factor
        
        % --- ORBITAL MECHANICS ---
        MU = 398600.4418;             % Standard Gravitational Parameter (km^3/s^2)
        J2 = 1.08262668e-3;           % Second Zonal Harmonic (J2 Perturbation)
        H_ATM = 80;                   % Atmospheric Interference Limit (km)
        
        % --- RF ENGINEERING (Ka-Band ISL) ---
        C_LIGHT = 299792.458;         % Speed of Light (km/s)
        FREQ_HZ = 26e9;               % Carrier Frequency (26 GHz)
        G_MAX_DBI = 38.5;             % Peak Antenna Gain (Phased Array)
        ANTENNA_N = 1.2;              % Cosine Roll-off Exponent (Beamwidth proxy)
        SYS_LOSS_DB = 3.0;            % System Noise / Cabling Losses
    end
    
    methods (Static)
        
        %% 1. ORBITAL GENERATION & PROPAGATION
        % =================================================================
        
		
        function [pos, vel, loads] = generateConstellation(N, congestionLevel, type)
            % GENERATECONSTELLATION - Generates satellite state vectors.
            % Supported modes: Walker-Delta (Organized) or Stochastic (Random).
            if nargin < 2, congestionLevel = 1.0; end
            if nargin < 3, type = 'Starlink'; end
            
            % Define Constellation Parameters
            switch type
                case 'Starlink'
                    h_sat = 550; inc_deg = 53;   
                case 'OneWeb'
                    h_sat = 1200; inc_deg = 87.9; 
                case 'Kuiper'
                    h_sat = 630; inc_deg = 51.9;  
                otherwise
                    h_sat = 550; inc_deg = 53;
            end
            
            R = SimUtils.R_EARTH + h_sat;
            v_mag = sqrt(SimUtils.MU / R); % Vis-Viva Equation (Circular)
            
			% --- STOCHASTIC MODE (Binomial Point Process) ---
            % Implements Inverse Transform Sampling on a Sphere to ensure
            % uniform density, avoiding polar clustering.
            if strcmp(type, 'Stochastic')
                pos = zeros(N, 3);
                vel = zeros(N, 3);
                loads = rand(N, 1) * congestionLevel;
                
                for i = 1:N
                    % 1. Random Point on Sphere (Archimedes' Hat-Box Theorem)
                    z = 2 * rand() - 1;       % Uniform z [-1, 1]
                    theta = 2 * pi * rand();  % Uniform longitude [0, 2pi]
                    r_xy = sqrt(1 - z^2);
                    
                    x = R * r_xy * cos(theta);
                    y = R * r_xy * sin(theta);
                    pos(i, :) = [x, y, z*R];
                    
                    % 2. Random Velocity (Tangent to sphere surface)                  
                    rand_v = randn(1,3);
                    v_tan = cross(pos(i,:), rand_v); 
                    vel(i, :) = (v_tan / norm(v_tan)) * v_mag;
                end
                return; 
            end
            % --------------------------------------------------
			
            % Walker-Delta Logic: T/P/F
            numPlanes = floor(sqrt(N));
            satsPerPlane = ceil(N / numPlanes);
            N_actual = numPlanes * satsPerPlane;
            
            pos = zeros(N_actual, 3); 
            vel = zeros(N_actual, 3);
            loads = rand(N_actual, 1) * congestionLevel;
            
            inc = deg2rad(inc_deg);
            d_RAAN = 2 * pi / numPlanes;        
            d_MeanAnomaly = 2 * pi / satsPerPlane; 
            
            idx = 1;
            for p = 0:numPlanes-1
                raan = p * d_RAAN;
                for s = 0:satsPerPlane-1
                    M = s * d_MeanAnomaly;
                    
                    % 1. Perifocal Frame
                    r_pqw = [R*cos(M); R*sin(M); 0];
                    v_pqw = [-v_mag*sin(M); v_mag*cos(M); 0];
                    
                    % 2. Rotation Matrices
                    R_inc = [1 0 0; 0 cos(inc) -sin(inc); 0 sin(inc) cos(inc)];
                    R_raan = [cos(raan) -sin(raan) 0; sin(raan) cos(raan) 0; 0 0 1];
                    
                    % 3. ECI Transformation
                    Q = R_raan * R_inc;
                    pos(idx,:) = (Q * r_pqw)';
                    vel(idx,:) = (Q * v_pqw)';
                    idx = idx + 1;
                end
            end
        end
        
        %% 2. TOPOLOGY & ROUTING LOGIC (WANG ET AL.)
        % =================================================================
        
        function [G_std, G_opt] = buildGraphs(pos, vel, loads, P)
            % BUILDGRAPHS - Constructs network topology and weighted graphs.
            % G_std: Standard Distance-based weights.
            % G_opt: "Effective Latency" weights (Wang et al.) considering reliability.
            
            N = size(pos, 1);
            
            % --- 1. Geometric Filtering (Range & Occlusion) ---
            D = squareform(pdist(pos)); 
            InRange = D <= P.Range & D > 1; 
            [row, col] = find(triu(InRange, 1));
            
            if isempty(row)
                G_std = graph(); G_opt = graph(); return; 
            end
            
			% Check Earth Occlusion
            P1 = pos(row, :); P2 = pos(col, :);
            CrossP = cross(P1, P2, 2); 
            Area = vecnorm(CrossP, 2, 2);
            Dist_Edge = D(sub2ind([N, N], row, col));
            H_min = Area ./ Dist_Edge; 
            
            ValidLoS = H_min > (SimUtils.R_EARTH + SimUtils.H_ATM);
            valid_idx = find(ValidLoS);
            r_v = row(valid_idx); c_v = col(valid_idx);
            
			% --- 2. Hardware Constraints (Degree Limit) ---
            % Enforce max K connections per satellite (e.g., 4 lasers)
            Adj = sparse(r_v, c_v, true, N, N); Adj = Adj | Adj';
            D_masked = D; D_masked(~Adj) = inf;
            [~, sort_idx] = sort(D_masked, 2, 'ascend');
            
            MAX_DEG = 4; 
            topK = sort_idx(:, 1:MAX_DEG);
            row_ids = repmat((1:N)', 1, MAX_DEG);
            valid_k = ~isinf(D_masked(sub2ind([N,N], row_ids, topK)));
            
            % Final Adjacency Matrix
            AdjConstrained = sparse(row_ids(valid_k), topK(valid_k), 1, N, N);
            AdjConstrained = max(AdjConstrained, AdjConstrained'); 

            % --- 3. Standard Graph (Pure Propagation Delay) ---
            % Weight = Distance (km)
            W_std = sparse(D) .* AdjConstrained;
            G_std = graph(W_std);
            
            % --- 4. Reliability-Aware Graph (Wang et al. Method) ---
            if isfield(P, 'UseOpt') && P.UseOpt
                [u, v] = find(triu(AdjConstrained, 1));
                
                % A. Calculate Physical Latency (T_prop)
                d_uv = zeros(length(u), 1);
                for k=1:length(u), d_uv(k) = D(u(k), v(k)); end
                T_prop = d_uv / SimUtils.C_LIGHT; % Seconds
                
                % B. Calculate Reliability (Outage Probability P_out)
				
				% --- Noise Calculation (Thermal Model) ---
                k_B = 1.380649e-23;   % Boltzmann constant (J/K)
                T_sys = 290;          % System Noise Temp (K)
                BW_Hz = 400e6;        % Bandwidth (400 MHz)
                Noise_Watts = k_B * T_sys * BW_Hz;
                Noise_dBW = 10 * log10(Noise_Watts);
				
                % Link Budget Parameters
                Tx_Power_dBW = 0; 
                G_Tx_dBi = 30; G_Rx_dBi = 30;
                Freq_Hz = 26e9; 
                c = 299792458;
                
                % FSPL & SNR
                d_meters = d_uv * 1000;
                fspl_val = (4 * pi * d_meters * Freq_Hz / c).^2;
                fspl_db = 10 * log10(fspl_val);
                
                rx_dbw = Tx_Power_dBW + G_Tx_dBi + G_Rx_dBi - fspl_db;
                snr_db = rx_dbw - Noise_dBW;
                snr_linear = 10.^(snr_db ./ 10);
                
                % Rayleigh Fading Outage Probability
                % P_out = 1 - exp(-Theta / SNR)
                Theta = 10^(5/10); % 5dB Threshold
                p_out = 1 - exp(-Theta ./ snr_linear);
                
                % Safety clamps for numerical stability
                p_out = max(p_out, 1e-9); 
                p_out = min(p_out, 0.999); 
                
                % C. EFFECTIVE LATENCY METRIC
                % Combines Propagation Delay, Reliability (ARQ), and Queueing.
                
            if isempty(loads)
                    load_penalty = zeros(size(u)); % No load data provided -> Zero penalty
                else
                    load_penalty = (loads(u) + loads(v)) / 2; 
                end
                
                % Queueing Model (Congestion)
                % Models the exponential delay increase as buffers fill up.
                % Approximation: T_queue ~ Load^2 * base_processing_time
                T_queue = (load_penalty .^ 2) * 0.005; 
                
                % Final Effective Latency Formula (Wang et al. modified)
                % L_eff = L_prop / (1 - P_out) + L_queue
                K_max = 5;
				E_N_tx = (1 - (K_max+1)*p_out^K_max + K_max*p_out^(K_max+1)) / (1-p_out)^2;
				T_effective = T_prop * E_N_tx + T_queue;

                % ----------------------------------------------------
                
                % Weighted Cost Function (User Tunable)
                w = P.Wang_Alpha / 100;
                Edge_Weight = (1 - w) * T_prop + w * T_effective;
                
                % Create Weighted Graph
                W_opt = sparse(u, v, Edge_Weight, N, N);
                W_opt = W_opt + W_opt'; 
                G_opt = graph(W_opt);
            else
                G_opt = G_std;
            end
        end
        
        %% 3. ANALYSIS METRICS (IMPROVED)
        % =================================================================
        
        function [lat, jitter, fspl_db, doppler_max] = getPathMetrics(path, pos, vel, ~, ~)
            % GETPATHMETRICS - Calculates detailed performance metrics for a route.
            % Returns: Latency (ms), Jitter (us), Path Loss (dB), Max Doppler (Hz).
            
            if length(path) < 2
                lat=NaN; jitter=NaN; fspl_db=NaN; doppler_max=NaN; return;
            end
            
            P_nodes = pos(path, :);
            V_nodes = vel(path, :);
            
            % Path Vectors
            segments = diff(P_nodes,1,1); % Vector d_uv
            dists = vecnorm(segments, 2, 2); % Scalar Distance
            
            % Metric 1: Physical Latency (ms)
            lat = sum(dists) / SimUtils.C_LIGHT * 1000;
            
            % Metric 2: Doppler Shift (Hz) & Jitter
            % Doppler Delta_f = (f/c) * (v_rel dot u_vec)
            vel_rel = diff(V_nodes, 1, 1); % V_rx - V_tx
            u_vec = segments ./ dists;     % Unit vector direction
            
            % Range Rate (Velocity along the link)
            range_rate = sum(vel_rel .* u_vec, 2); % km/s
            
           % --- DOPPLER & JITTER PHYSICS ---
			f_c = SimUtils.FREQ_HZ;
			c_km = SimUtils.C_LIGHT;

			% 1. Doppler Shift (RF Layer)
			doppler_shifts = (f_c / c_km) * range_rate; % Hz
			doppler_max = max(abs(doppler_shifts));

			% 2. Network Jitter (Packet Delay Variation)
            % Jitter is the rate of change of latency multiplied by packet interval.
			path_length_rate = sum(range_rate);
			latency_drift = path_length_rate / c_km;
			
			% Assume packet flow every 10ms (Real-Time Application)
			T_interval = 0.010;
			
			% Jitter 
			lat_per_hop = vecnorm(diff(pos(path,:),1,1), 2, 2) / 299792.458;
			jitter = std(lat_per_hop) * 1e6;

          
            % Metric 3: Link Budget
            lambda = (c_km * 1000) / f_c;
            d_m = dists * 1000;
            L_fspl = (4 * pi * d_m / lambda).^2;
            L_fspl_db = 10 * log10(L_fspl);
            
			% --- Antenna Gain (Steerable / Phased Array Model) ---           
            POINTING_LOSS_DB = 1.0; % Typical misalignment loss
            
            % Gain (Tx και Rx)
            G_tx_db = SimUtils.G_MAX_DBI - POINTING_LOSS_DB;
            G_rx_db = SimUtils.G_MAX_DBI - POINTING_LOSS_DB;
            
            
            Hop_Loss_dB = L_fspl_db - G_tx_db - G_rx_db + SimUtils.SYS_LOSS_DB;
            fspl_db = mean(Hop_Loss_dB);
        end
        
        %% 4. UTILITIES
        % =================================================================
        
        function ecef = lla2ecef(lla)
            % lla2ecef - Geodetic to Cartesian conversion
            lat = deg2rad(lla(1)); lon = deg2rad(lla(2)); alt = 0;
            
            a = SimUtils.R_EARTH; 
            f = SimUtils.FLATTENING; 
            e2 = 2*f - f^2;
            
            N = a / sqrt(1 - e2 * sin(lat)^2);
            
            x = (N + alt) * cos(lat) * cos(lon);
            y = (N + alt) * cos(lat) * sin(lon);
            z = (N * (1 - e2) + alt) * sin(lat);
            ecef = [x, y, z];
        end
    end
end
