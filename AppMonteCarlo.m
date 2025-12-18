classdef SimUtils
    % SIMUTILS - High-Fidelity LEO Astrodynamics & Graph Theory Engine
    % DESCRIPTION: 
    %   Updated for Wang et al. (2024) Reliability-Aware Routing.
    %     1. "Effective Latency" Metric (Latency / (1 - P_out)) [Wang Eq. 24]
    %     2. Explicit Doppler Shift Calculation (for Jitter analysis)
    %     3. Accurate Link Budgeting for Outage Probability
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
            % generateConstellation - Creates initial state vectors (Walker-Delta)
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
        
        %% 2. TOPOLOGY & ROUTING LOGIC (IMPROVED)
        % =================================================================
        
        function [G_std, G_opt] = buildGraphs(pos, vel, loads, P)
            % buildGraphs - Constructs Topology using Effective Latency (ARQ)
            %
            
            N = size(pos, 1);
            
            % --- 1. Geometric Filtering (Range & Occlusion) ---
            D = squareform(pdist(pos)); 
            InRange = D <= P.Range & D > 1; 
            [row, col] = find(triu(InRange, 1));
            
            if isempty(row)
                G_std = graph(); G_opt = graph(); return; 
            end
            
            P1 = pos(row, :); P2 = pos(col, :);
            CrossP = cross(P1, P2, 2); 
            Area = vecnorm(CrossP, 2, 2);
            Dist_Edge = D(sub2ind([N, N], row, col));
            H_min = Area ./ Dist_Edge; 
            
            ValidLoS = H_min > (SimUtils.R_EARTH + SimUtils.H_ATM);
            valid_idx = find(ValidLoS);
            r_v = row(valid_idx); c_v = col(valid_idx);
            
            % --- 2. Hardware Constraints (Degree Limit) ---
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
                % Link Budget Parameters
                Tx_Power_dBW = 0; 
                G_Tx_dBi = 30; G_Rx_dBi = 30;
                Noise_dBW = -120; 
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
                
                % Safety clamps
                p_out = max(p_out, 1e-9); 
                p_out = min(p_out, 0.999); % Prevent division by zero
                
                % C. EFFECTIVE LATENCY (The "Wang" Metric)
                % T_eff = T_prop + (Retransmissions * T_round_trip)
                % Simplified as Expectation: T_eff = T_prop / (1 - P_out)
                % This couples Reliability and Latency physically.
                
                % Apply User Weighting if desired (blending pure physics vs user pref)
                % Note: P.Wang_Alpha is 0-100.
                % If Alpha=0: Pure Distance. If Alpha=100: Pure Effective Latency.
                w = P.Wang_Alpha / 100;
                
                % Metric: Weighted sum of Pure Latency and Risk-Adjusted Latency
                % (This allows the slider to still function as a trade-off explorer)
                T_effective = T_prop ./ (1 - p_out);
                
                Edge_Weight = (1 - w) * T_prop + w * T_effective;
                
                % Create Graph
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
            % getPathMetrics - Calculates detailed link performance
            
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
			% Calculation of frequency shift for each hop
			doppler_shifts = (f_c / c_km) * range_rate; % Hz
			doppler_max = max(abs(doppler_shifts));

			% 2. Network Jitter (Packet Delay Variation)
			% Jitter is not the std of Doppler. It is the rate of change
			% of latency (dLatency/dt) times the packet time interval.
			% Rate of change of path length (km/s)
			
			path_length_rate = sum(range_rate);
			% Rate of change of latency (s/s - dimensionless)
			latency_drift = path_length_rate / c_km;
			
			% Assume packet flow every 10ms (typical for Real-Time applications)
			T_interval = 0.010;
			
			% Jitter = How much the latency changed between two packets
			% Convert to Microseconds (us) for readability
			jitter = abs(latency_drift * T_interval) * 1e6;

            
            % Metric 3: Link Budget
            lambda = (c_km * 1000) / f_c;
            d_m = dists * 1000;
            L_fspl = (4 * pi * d_m / lambda).^2;
            L_fspl_db = 10 * log10(L_fspl);
            
			% --- Antenna Gain (Steerable / Phased Array Model) ---
            
            POINTING_LOSS_DB = 1.0; % typical misalignment loss value (1-2 dB)
            
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
