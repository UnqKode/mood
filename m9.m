%Task 1: Blackman Window Based Low-pass and High-pass FIR Filter
clc;
clear all;
close all;
M = 31;
N = 0:M-1;
wc = [pi/2, pi/4, pi/6];
wc2 = pi/4;
w = 0.42 - 0.5*cos(2*pi*N/(M-1)) + 0.08*cos(4*pi*N/(M-1));

figure;

for i=1:length(wc)
    hd_low = sin(wc(i)*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_low2 = sin(wc2*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_high = sin(pi*(N-(M-1)/2))./(pi*(N-(M-1)/2)) - hd_low;


    hd_low((M+1)/2) = wc(i)/pi;
    hd_high((M+1)/2) = 1 - (wc(i)/pi);
    


    hn_low = hd_low.*w;
    hn_high = hd_high.*w;
    
    
    subplot(3, 2, 2*i-1);
    [H_low, omega] = freqz(hn_low, 1, 512);
    plot(omega/pi, 20*log10(abs(H_low)));
    title(['Low-Pass Filter, wc = ', num2str(wc(i)/pi), '\pi']);
    grid on;
    
    subplot(3, 2, 2*i);
    [H_high, omega] = freqz(hn_high, 1, 512);
    plot(omega/pi, 20*log10(abs(H_high)));
    title(['High-Pass Filter, wc = ', num2str(wc(i)/pi), '\pi']);
    grid on;
    
end

%Task 2: Hamming Window Based Low-pass and High-pass FIR Filter
clc;
clear all;
close all;
M = 31;
N = 0:M-1;
wc = [pi/2, pi/4, pi/6];
wc2 = pi/4;
w = 0.54 - 0.46*cos(2*pi*N/(M-1));

figure;

for i=1:length(wc)
    hd_low = sin(wc(i)*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_low2 = sin(wc2*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_high = sin(pi*(N-(M-1)/2))./(pi*(N-(M-1)/2)) - hd_low;


    hd_low((M+1)/2) = wc(i)/pi;
    hd_high((M+1)/2) = 1 - (wc(i)/pi);
    


    hn_low = hd_low.*w;
    hn_high = hd_high.*w;
    
    
    subplot(3, 2, 2*i-1);
    [H_low, omega] = freqz(hn_low, 1, 512);
    plot(omega/pi, 20*log10(abs(H_low)));
    title(['Low-Pass Filter, wc = ', num2str(wc(i)/pi), '\pi']);
    grid on;
    
    subplot(3, 2, 2*i);
    [H_high, omega] = freqz(hn_high, 1, 512);
    plot(omega/pi, 20*log10(abs(H_high)));
    title(['High-Pass Filter, wc = ', num2str(wc(i)/pi), '\pi']);
    grid on;
    
end

%Task 3: Rectangular Window Based Low-pass and High-pass FIR Filter
clc;
clear all;
close all;
M = 31;
N = 0:M-1;
wc = [pi/2, pi/4, pi/6];
wc2 = pi/4;
w = 1;
figure;
for i=1:length(wc)
    hd_low = sin(wc(i)*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_low2 = sin(wc2*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_high = sin(pi*(N-(M-1)/2))./(pi*(N-(M-1)/2)) - hd_low;
    hd_low((M+1)/2) = wc(i)/pi;
    hd_high((M+1)/2) = 1 - (wc(i)/pi);
    hn_low = hd_low.*w;
    hn_high = hd_high.*w;
    
    subplot(3, 2, 2*i-1);
    [H_low, omega] = freqz(hn_low, 1, 512);
    plot(omega/pi, 20*log10(abs(H_low)));
    title(['Low-Pass Filter, wc = ', num2str(wc(i)/pi), '\pi']);
    grid on;

    subplot(3, 2, 2*i);
    [H_high, omega] = freqz(hn_high, 1, 512);
    plot(omega/pi, 20*log10(abs(H_high)));
    title(['High-Pass Filter, wc = ', num2str(wc(i)/pi), '\pi']);
    grid on;
    
end

%Task 4: Blackman Window Based BP and BS FIR Filter
clc;
clear all;
close all;
M = 31;
N = 0:M-1;
wc = pi/2;
wc2 = pi/4;
w = 0.42 - 0.5*cos(2*pi*N/(M-1)) + 0.08*cos(4*pi*N/(M-1));
for i=1:2
    hd_low = sin(wc*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_low2 = sin(wc2*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_bp = hd_low - hd_low2;
    hd_bs = sin(pi*(N-(M-1)/2))./(pi*(N-(M-1)/2)) - hd_bp;
    hd_low((M+1)/2) = wc/pi;
    hd_high((M+1)/2) = 1 - (wc/pi);
    hd_bp((M+1)/2) = wc/pi;
    hd_bs((M+1)/2) = 1 - ((wc-wc2)/pi);
    hn_bp = hd_bp.*w;
    hn_bs = hd_bs.*w;
    
    if i == 1
        figure(1);
        freqz(hn_bp,1,512);
    else
        figure(2);
        freqz(hn_bs,1,512);
    end
end

%Task 5: Hamming Window Based BP and BS FIR Filter
clc;
clear all;
close all;
M = 31;
N = 0:M-1;
wc = pi/2;
wc2 = pi/4;
w = 0.54 - 0.46*cos(2*pi*N/(M-1));
for i=1:2
    hd_low = sin(wc*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_low2 = sin(wc2*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_bp = hd_low - hd_low2;
    hd_bs = sin(pi*(N-(M-1)/2))./(pi*(N-(M-1)/2)) - hd_bp;
    hd_low((M+1)/2) = wc/pi;
    hd_high((M+1)/2) = 1 - (wc/pi);
    hd_bp((M+1)/2) = wc/pi;
    hd_bs((M+1)/2) = 1 - ((wc-wc2)/pi);
    hn_bp = hd_bp.*w;
    hn_bs = hd_bs.*w;
    
    if i == 1
        figure(1);
        freqz(hn_bp,1,512);
    else
        figure(2);
        freqz(hn_bs,1,512);
    end
end

%Task 6: Rectangular Window Based BP and BS FIR Filter
clc;
clear all;
close all;
M = 31;
N = 0:M-1;
wc = pi/2;
wc2 = pi/4;
w = 1;
for i=1:2
    hd_low = sin(wc*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_low2 = sin(wc2*(N-(M-1)/2))./(pi*(N-(M-1)/2));
    hd_bp = hd_low - hd_low2;
    hd_bs = sin(pi*(N-(M-1)/2))./(pi*(N-(M-1)/2)) - hd_bp;
    hd_low((M+1)/2) = wc/pi;
    hd_high((M+1)/2) = 1 - (wc/pi);
    hd_bp((M+1)/2) = wc/pi;
    hd_bs((M+1)/2) = 1 - ((wc-wc2)/pi);
    hn_bp = hd_bp.*w;
    hn_bs = hd_bs.*w;
    
    if i == 1
        figure(1);
        freqz(hn_bp,1,512);
    else
        figure(2);
        freqz(hn_bs,1,512);
    end
end

%Task 7: Hiss generation and removal using Blackman Window
clc;
close all;
clear all;

% Read audio file
[xin, fs] = audioread("handel.wav");

% Trim to multiple of 256 samples for proper block processing
num_blocks = floor(length(xin)/256);
xin = xin(1:num_blocks*256);

% Initialize output array
Xiff_out = zeros(size(xin));

% Create hiss in frequency domain
hiss = zeros(256, 1);
hiss(66, 1) = 7;
hiss(192, 1) = 7;

% Process in blocks of 256 samples
for i = 1:256:length(xin)
    block_end = min(i+255, length(xin));
    current_block = xin(i:block_end);
    
    % Ensure block is exactly 256 samples (pad if necessary)
    if length(current_block) < 256
        current_block = [current_block; zeros(256-length(current_block), 1)];
    end
    
    % Apply FFT, add hiss, and inverse FFT
    Xout = fft(current_block) + hiss;
    Xiff_out(i:i+255) = real(ifft(Xout)); % Take real part to avoid complex values
end

% Write output file
audiowrite("Out+Hiss.wav", Xiff_out, fs);

% Design FIR filter
M = 31;
n = 0:M-1;
w_c = pi/2;

% Blackman window
w = 0.42 - 0.5 * cos((2*pi*n)/(M-1)) + 0.08 * cos((4*pi*n)/(M-1));

% Ideal low-pass filter coefficients
h_d = sin(w_c * (n - (M-1)/2)) ./ (pi * (n - (M-1)/2));
h_d(isnan(h_d)) = w_c/pi; % Handle division by zero at center point

% Apply window to get final filter coefficients
h = h_d .* w;

% Read the noisy audio
[x, fs] = audioread("Out+Hiss.wav");

% Apply filter
y = filter(h, 1, x);

% Normalize audio to prevent clipping
Xiff_out = Xiff_out / max(abs(Xiff_out));
y = y / max(abs(y));

% Play sounds
disp('Playing noisy audio...');
sound(Xiff_out, fs);
pause(10);

disp('Playing filtered audio...');
sound(y, fs);

% Optional: Plot frequency response
figure;
freqz(h, 1, 512);
title('Filter Frequency Response');

% Optional: Plot original and filtered signals in time domain
figure;
subplot(3,1,1);
plot(xin(1:min(5000, length(xin))));
title('Original Signal');
xlabel('Samples');
ylabel('Amplitude');

subplot(3,1,2);
plot(Xiff_out(1:min(5000, length(Xiff_out))));
title('Noisy Signal (with Hiss)');
xlabel('Samples');
ylabel('Amplitude');

subplot(3,1,3);
plot(y(1:min(5000, length(y))));
title('Filtered Signal');
xlabel('Samples');
ylabel('Amplitude');

%Task 8: Additional Hiss added to audio
clc;
clear all;
close all;

% Read audio file
[xin, Fs] = audioread("handel.wav");

% Trim to multiple of 8 samples for proper block processing
num_blocks = floor(length(xin)/8);
xin = xin(1:num_blocks*8);

% Initialize output array with proper dimensions
xiffout = zeros(size(xin));

% Create hiss in frequency domain (affecting specific frequency bins)
hiss = zeros(8, 1);
hiss(3, 1) = 3;  % Affects bin 3
hiss(7, 1) = 3;  % Affects bin 7

% Process in blocks of 8 samples
for i = 1:8:length(xin)
    block_end = i+7;
    
    % Ensure we don't exceed array bounds
    if block_end > length(xin)
        break;
    end
    
    % Apply FFT, add hiss, and inverse FFT
    X_fft = fft(xin(i:block_end));
    Xout = X_fft + hiss;
    xiffout(i:block_end) = real(ifft(Xout)); % Take real part
end

% Write output file
audiowrite("output+hiss.wav", xiffout, Fs);

% Read the noisy audio
[x, fs] = audioread("output+hiss.wav");

% Define filter coefficients (band-stop/band-pass filter)
hn = [1, -2*cos(pi/4), 1]; % Using pi/4 instead of pi/2 for better frequency response

% Analyze filter frequency response
figure;
freqz(hn, 1, 512);
title('Filter Frequency Response');

% Apply filtering
y = filter(hn, 1, x);

% Normalize audio to prevent clipping
xiffout = xiffout / max(abs(xiffout));
y = y / max(abs(y));

% Play sounds
disp('Playing noisy audio (with hiss)...');
sound(xiffout, fs);
pause(10);

disp('Playing filtered audio...');
sound(y, fs);

% Optional: Plot signals for comparison
figure;
subplot(3,1,1);
plot(xin(1:min(2000, length(xin))));
title('Original Signal');
xlabel('Samples');
ylabel('Amplitude');
grid on;

subplot(3,1,2);
plot(xiffout(1:min(2000, length(xiffout))));
title('Noisy Signal (with Hiss)');
xlabel('Samples');
ylabel('Amplitude');
grid on;

subplot(3,1,3);
plot(y(1:min(2000, length(y))));
title('Filtered Signal');
xlabel('Samples');
ylabel('Amplitude');
grid on;

% Optional: Frequency domain analysis
figure;
N = min(1024, length(xin));
f = (0:N-1)*fs/N;
subplot(2,1,1);
X_orig = abs(fft(xin(1:N), N));
plot(f(1:N/2), X_orig(1:N/2));
title('Original Signal Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

subplot(2,1,2);
X_noisy = abs(fft(xiffout(1:N), N));
plot(f(1:N/2), X_noisy(1:N/2));
title('Noisy Signal Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
