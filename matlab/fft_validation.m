% filepath: /Users/P2956632/Documents/CS 3120/Music_Feature_Analysis/matlab/fft_validation.m
% FFT Validation for Music Feature Analysis
% Validates FFT-based feature extraction and Parseval's theorem

function fft_validation()
    fprintf('======================================================================\n');
    fprintf('FFT VALIDATION - Music Feature Analysis\n');
    fprintf('======================================================================\n\n');
    
    % Test 1: Parseval's Theorem
    test_parsevals_theorem();
    
    % Test 2: Spectral Centroid
    test_spectral_centroid();
    
    % Test 3: Spectral Rolloff
    test_spectral_rolloff();
    
    % Test 4: Windowing Effects
    test_windowing_effects();
    
    fprintf('\n======================================================================\n');
    fprintf('FFT VALIDATION COMPLETE\n');
    fprintf('======================================================================\n');
end

function test_parsevals_theorem()
    fprintf('Test 1: Parseval''s Theorem Validation\n');
    fprintf('----------------------------------------------------------------------\n');
    
    % Generate test signal
    fs = 44100;  % Sample rate
    duration = 1;
    t = 0:1/fs:duration-1/fs;
    test_signal = sin(2*pi*440*t) + 0.5*sin(2*pi*880*t);
    
    % Time-domain energy
    time_energy = sum(test_signal.^2);
    
    % Frequency-domain energy
    fft_mag = abs(fft(test_signal));
    freq_energy = sum(fft_mag.^2) / length(test_signal);
    
    % Calculate error
    error = abs(time_energy - freq_energy) / time_energy * 100;
    
    fprintf('  Time-domain energy:      %.6f\n', time_energy);
    fprintf('  Frequency-domain energy: %.6f\n', freq_energy);
    fprintf('  Error:                   %.6f%%\n', error);
    
    if error < 1
        fprintf('  Result: PASSED (error < 1%%)\n');
    else
        fprintf('  Result: FAILED (error >= 1%%)\n');
    end
    fprintf('\n');
end

function test_spectral_centroid()
    fprintf('Test 2: Spectral Centroid Computation\n');
    fprintf('----------------------------------------------------------------------\n');
    
    fs = 44100;
    duration = 1;
    t = 0:1/fs:duration-1/fs;
    
    % Low-frequency signal (100 Hz)
    low_freq_signal = sin(2*pi*100*t);
    
    % Compute FFT
    fft_mag = abs(fft(low_freq_signal));
    freqs = (0:length(fft_mag)-1) * fs / length(fft_mag);
    
    % Positive frequencies only
    half_idx = floor(length(fft_mag)/2);
    freqs_pos = freqs(1:half_idx);
    fft_mag_pos = fft_mag(1:half_idx);
    
    % Compute spectral centroid
    centroid = sum(fft_mag_pos .* freqs_pos') / sum(fft_mag_pos);
    
    fprintf('  Expected centroid: ~100 Hz\n');
    fprintf('  Computed centroid: %.2f Hz\n', centroid);
    
    if centroid >= 90 && centroid <= 110
        fprintf('  Result: PASSED (within ±10 Hz)\n');
    else
        fprintf('  Result: FAILED (outside expected range)\n');
    end
    fprintf('\n');
end

function test_spectral_rolloff()
    fprintf('Test 3: Spectral Rolloff (95th percentile)\n');
    fprintf('----------------------------------------------------------------------\n');
    
    fs = 44100;
    duration = 1;
    t = 0:1/fs:duration-1/fs;
    
    % Test signal (1000 Hz)
    test_signal = sin(2*pi*1000*t);
    
    % Compute FFT
    fft_mag = abs(fft(test_signal));
    freqs = (0:length(fft_mag)-1) * fs / length(fft_mag);
    
    % Positive frequencies
    half_idx = floor(length(fft_mag)/2);
    freqs_pos = freqs(1:half_idx);
    fft_mag_pos = fft_mag(1:half_idx);
    
    % Cumulative sum of power
    cumsum_power = cumsum(fft_mag_pos);
    total_power = cumsum_power(end);
    
    % Find 95th percentile
    rolloff_idx = find(cumsum_power >= 0.95 * total_power, 1, 'first');
    rolloff_freq = freqs_pos(rolloff_idx);
    
    fprintf('  Expected rolloff: ~1000 Hz\n');
    fprintf('  Computed rolloff: %.2f Hz\n', rolloff_freq);
    
    if rolloff_freq >= 900
        fprintf('  Result: PASSED (rolloff > 900 Hz)\n');
    else
        fprintf('  Result: FAILED (rolloff too low)\n');
    end
    fprintf('\n');
end

function test_windowing_effects()
    fprintf('Test 4: Windowing Effects (Hann vs Rectangular)\n');
    fprintf('----------------------------------------------------------------------\n');
    
    fs = 44100;
    duration = 1;
    t = 0:1/fs:duration-1/fs;
    
    % Two sine waves
    test_signal = sin(2*pi*440*t) + 0.5*sin(2*pi*880*t);
    
    % Rectangular window (no window)
    fft_rect = abs(fft(test_signal));
    
    % Hann window
    window_hann = hann(length(test_signal))';
    fft_hann = abs(fft(test_signal .* window_hann));
    
    % Find peaks
    half_idx = floor(length(fft_rect)/2);
    
    [~, peaks_rect] = findpeaks(fft_rect(1:half_idx), 'MinPeakHeight', max(fft_rect(1:half_idx))*0.1);
    [~, peaks_hann] = findpeaks(fft_hann(1:half_idx), 'MinPeakHeight', max(fft_hann(1:half_idx))*0.1);
    
    fprintf('  Peaks found (rectangular): %d\n', length(peaks_rect));
    fprintf('  Peaks found (Hann):        %d\n', length(peaks_hann));
    
    if length(peaks_rect) >= 2 && length(peaks_hann) >= 2
        fprintf('  Result: PASSED (both windows detected peaks)\n');
    else
        fprintf('  Result: FAILED (insufficient peaks detected)\n');
    end
    fprintf('\n');
end