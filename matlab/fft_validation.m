% filepath: /Users/P2956632/Documents/CS 3120/Music_Feature_Analysis/matlab/fft_validation.m

%% FFT Validation: Comparing Mathematical Analysis with ML Features
% This script validates that ML-extracted features correlate with
% FFT-based numerical analysis, ensuring consistency between approaches.

clear all; close all; clc;

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('FFT VALIDATION: Numerical Analysis vs Machine Learning Features\n');
fprintf('%s\n\n', repmat('=', 1, 70));

%% 1. Generate Test Signals with Known Properties
fprintf('1. Generating test signals with known spectral properties...\n');

fs = 44100;  % Sampling frequency
duration = 1;  % 1 second
t = 0:1/fs:duration-1/fs;

% Test signals: vary frequency content systematically
test_signals = struct();

% Low-frequency dominant signal (e.g., bass music)
test_signals(1).name = 'Low-Frequency';
test_signals(1).signal = sin(2*pi*100*t) + 0.5*sin(2*pi*50*t) + 0.1*randn(size(t));
test_signals(1).expected_centroid = 100;  % Approximate expected centroid

% Mid-frequency dominant signal
test_signals(2).name = 'Mid-Frequency';
test_signals(2).signal = sin(2*pi*500*t) + 0.5*sin(2*pi*700*t) + 0.1*randn(size(t));
test_signals(2).expected_centroid = 600;

% High-frequency dominant signal
test_signals(3).name = 'High-Frequency';
test_signals(3).signal = sin(2*pi*5000*t) + 0.5*sin(2*pi*3000*t) + 0.1*randn(size(t));
test_signals(3).expected_centroid = 4000;

% Broadband signal (white noise)
test_signals(4).name = 'Broadband';
test_signals(4).signal = randn(size(t));
test_signals(4).expected_centroid = fs/4;  % Expected near Nyquist/2

fprintf('   Generated 4 test signals with varying frequency content\n\n');

%% 2. Compute FFT Features for Each Signal
fprintf('2. Computing FFT-based features for each test signal...\n');

N = length(test_signals(1).signal);
freq = (0:N-1) * fs / N;
positive_freq_idx = 1:floor(N/2);
freq_positive = freq(positive_freq_idx);

% Compute features for each signal
for i = 1:length(test_signals)
    fft_mag = abs(fft(test_signals(i).signal));
    fft_mag_pos = fft_mag(positive_freq_idx);
    
    % Spectral centroid
    centroid = sum(fft_mag_pos .* freq_positive) / sum(fft_mag_pos);
    test_signals(i).fft_centroid = centroid;
    
    % Spectral spread
    spread = sqrt(sum(fft_mag_pos .* (freq_positive - centroid).^2) / sum(fft_mag_pos));
    test_signals(i).fft_spread = spread;
    
    % Spectral rolloff (95th percentile)
    cumsum_power = cumsum(fft_mag_pos);
    total_power = cumsum_power(end);
    rolloff_idx = find(cumsum_power >= 0.95 * total_power, 1);
    test_signals(i).fft_rolloff = freq_positive(rolloff_idx);
end

fprintf('   FFT features computed for all signals\n\n');

%% 3. Create Comparison Table
fprintf('3. FFT Feature Validation Results:\n\n');

validation_table = table();
for i = 1:length(test_signals)
    validation_table = [validation_table; {
        test_signals(i).name, ...
        test_signals(i).expected_centroid, ...
        test_signals(i).fft_centroid, ...
        abs(test_signals(i).expected_centroid - test_signals(i).fft_centroid) / test_signals(i).expected_centroid * 100, ...
        test_signals(i).fft_spread, ...
        test_signals(i).fft_rolloff
    }];
end

validation_table.Properties.VariableNames = {...
    'Signal', 'Expected_Centroid_Hz', 'Computed_Centroid_Hz', 'Error_Percent', 'Spread_Hz', 'Rolloff_Hz'};

disp(validation_table);
fprintf('\n');

%% 4. Verify FFT Properties (Parseval's Theorem)
fprintf('4. Validating FFT properties (Parseval''s Theorem)...\n');

signal = test_signals(1).signal;
fft_mag = abs(fft(signal));

% Time-domain energy
time_energy = sum(signal.^2) / length(signal);

% Frequency-domain energy (Parseval's Theorem)
freq_energy = sum(fft_mag.^2) / (length(signal)^2);

% Normalized frequency energy
freq_energy_normalized = sum(fft_mag.^2) / length(signal);

error_parseval = abs(time_energy - freq_energy_normalized) / time_energy * 100;

fprintf('   Time-domain energy: %.4f\n', time_energy);
fprintf('   Frequency-domain energy (Parseval): %.4f\n', freq_energy_normalized);
fprintf('   Error: %.2f%%\n\n', error_parseval);

if error_parseval < 1
    fprintf('   Parseval''s Theorem validated (error < 1%%)\n');
else
    fprintf('   Parseval''s Theorem error exceeds 1%%\n');
end

fprintf('\n');

%% 5. Windowing Effects Demonstration
fprintf('5. Demonstrating windowing effects on FFT...\n');

test_signal = sin(2*pi*440*t) + 0.5*sin(2*pi*880*t);  % Two sine waves

% Rectangular window (no window)
fft_rect = abs(fft(test_signal));
fft_rect_pos = fft_rect(positive_freq_idx);

% Hann window
window_hann = hann(length(test_signal))';
fft_hann = abs(fft(test_signal .* window_hann));
fft_hann_pos = fft_hann(positive_freq_idx);% filepath: /Users/P2956632/Documents/CS 3120/Music_Feature_Analysis/matlab/fft_validation.m

%% FFT Validation: Comparing Mathematical Analysis with ML Features
% This script validates that ML-extracted features correlate with
% FFT-based numerical analysis, ensuring consistency between approaches.

clear all; close all; clc;

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('FFT VALIDATION: Numerical Analysis vs Machine Learning Features\n');
fprintf('%s\n\n', repmat('=', 1, 70));

%% 1. Generate Test Signals with Known Properties
fprintf('1. Generating test signals with known spectral properties...\n');

fs = 44100;  % Sampling frequency
duration = 1;  % 1 second
t = 0:1/fs:duration-1/fs;

% Test signals: vary frequency content systematically
test_signals = struct();

% Low-frequency dominant signal (e.g., bass music)
test_signals(1).name = 'Low-Frequency';
test_signals(1).signal = sin(2*pi*100*t) + 0.5*sin(2*pi*50*t) + 0.1*randn(size(t));
test_signals(1).expected_centroid = 100;  % Approximate expected centroid

% Mid-frequency dominant signal
test_signals(2).name = 'Mid-Frequency';
test_signals(2).signal = sin(2*pi*500*t) + 0.5*sin(2*pi*700*t) + 0.1*randn(size(t));
test_signals(2).expected_centroid = 600;

% High-frequency dominant signal
test_signals(3).name = 'High-Frequency';
test_signals(3).signal = sin(2*pi*5000*t) + 0.5*sin(2*pi*3000*t) + 0.1*randn(size(t));
test_signals(3).expected_centroid = 4000;

% Broadband signal (white noise)
test_signals(4).name = 'Broadband';
test_signals(4).signal = randn(size(t));
test_signals(4).expected_centroid = fs/4;  % Expected near Nyquist/2

fprintf('   Generated 4 test signals with varying frequency content\n\n');

%% 2. Compute FFT Features for Each Signal
fprintf('2. Computing FFT-based features for each test signal...\n');

N = length(test_signals(1).signal);
freq = (0:N-1) * fs / N;
positive_freq_idx = 1:floor(N/2);
freq_positive = freq(positive_freq_idx);

% Compute features for each signal
for i = 1:length(test_signals)
    fft_mag = abs(fft(test_signals(i).signal));
    fft_mag_pos = fft_mag(positive_freq_idx);
    
    % Spectral centroid
    centroid = sum(fft_mag_pos .* freq_positive) / sum(fft_mag_pos);
    test_signals(i).fft_centroid = centroid;
    
    % Spectral spread
    spread = sqrt(sum(fft_mag_pos .* (freq_positive - centroid).^2) / sum(fft_mag_pos));
    test_signals(i).fft_spread = spread;
    
    % Spectral rolloff (95th percentile)
    cumsum_power = cumsum(fft_mag_pos);
    total_power = cumsum_power(end);
    rolloff_idx = find(cumsum_power >= 0.95 * total_power, 1);
    test_signals(i).fft_rolloff = freq_positive(rolloff_idx);
end

fprintf('   FFT features computed for all signals\n\n');

%% 3. Create Comparison Table
fprintf('3. FFT Feature Validation Results:\n\n');

validation_table = table();
for i = 1:length(test_signals)
    validation_table = [validation_table; {
        test_signals(i).name, ...
        test_signals(i).expected_centroid, ...
        test_signals(i).fft_centroid, ...
        abs(test_signals(i).expected_centroid - test_signals(i).fft_centroid) / test_signals(i).expected_centroid * 100, ...
        test_signals(i).fft_spread, ...
        test_signals(i).fft_rolloff
    }];
end

validation_table.Properties.VariableNames = {...
    'Signal', 'Expected_Centroid_Hz', 'Computed_Centroid_Hz', 'Error_Percent', 'Spread_Hz', 'Rolloff_Hz'};

disp(validation_table);
fprintf('\n');

%% 4. Verify FFT Properties (Parseval's Theorem)
fprintf('4. Validating FFT properties (Parseval''s Theorem)...\n');

signal = test_signals(1).signal;
fft_mag = abs(fft(signal));

% Time-domain energy
time_energy = sum(signal.^2) / length(signal);

% Frequency-domain energy (Parseval's Theorem)
freq_energy = sum(fft_mag.^2) / (length(signal)^2);

% Normalized frequency energy
freq_energy_normalized = sum(fft_mag.^2) / length(signal);

error_parseval = abs(time_energy - freq_energy_normalized) / time_energy * 100;

fprintf('   Time-domain energy: %.4f\n', time_energy);
fprintf('   Frequency-domain energy (Parseval): %.4f\n', freq_energy_normalized);
fprintf('   Error: %.2f%%\n\n', error_parseval);

if error_parseval < 1
    fprintf('   Parseval''s Theorem validated (error < 1%%)\n');
else
    fprintf('   Parseval''s Theorem error exceeds 1%%\n');
end

fprintf('\n');

%% 5. Windowing Effects Demonstration
fprintf('5. Demonstrating windowing effects on FFT...\n');

test_signal = sin(2*pi*440*t) + 0.5*sin(2*pi*880*t);  % Two sine waves

% Rectangular window (no window)
fft_rect = abs(fft(test_signal));
fft_rect_pos = fft_rect(positive_freq_idx);

% Hann window
window_hann = hann(length(test_signal))';
fft_hann = abs(fft(test_signal .* window_hann));
fft_hann_pos = fft_hann(positive_freq_idx);