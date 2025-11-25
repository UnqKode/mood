
% Task 1: Continuous-time like sinusoid (high-rate sampled)
clc;
clear all;
close all;
A = 0.5; % Amplitude (1 V p-p)
fo = 3000; % Signal frequency (Hz)
fs = 1e5; % High "continuous-time-like" sampling frequency
n_cycles = 4; % Number of cycles
T0 = 1/fo; % Time period
t = 0 : 1/fs : n_cycles*T0;
x = A * cos(2*pi*fo*t);
figure;
plot(t, x, 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title('Task 1: x(t) = A cos(2\pi f_0 t), 4 cycles, F_s = 100 kHz');

% Task 2: Discrete-time representation using stem
clc;
clear all;
close all;
A = 0.5;
fo = 3000;
fs = 1e5;
n_cycles = 4;
T0 = 1/fo;
t = 0 : 1/fs : n_cycles*T0;
x = A * cos(2*pi*fo*t);
figure;
stem(t, x, 'filled');
grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title('Task 2: Discrete-time samples of x(t) at F_s = 100 kHz');

% Task 3: Sampling at various Fs and time-domain stem plots
clc;
clear all;
close all;
A = 0.5;
fo = 3000;
fs_ideal = 1e5;
n = 4;
T0 = 1/fo;
dur = n*T0; % Duration covering 4 cycles
t_ideal = 0 : 1/fs_ideal : dur; % High-rate reference
x_ideal = A * cos(2*pi*fo*t_ideal);
fs_list = [1e5 1e4 6e3 12e3 4e3 5e3]; % Includes ideal + given Fs
figure;
for k = 1:numel(fs_list)
fs_samp = fs_list(k);
ts = 0 : 1/fs_samp : dur;
y = A * cos(2*pi*fo*ts);
subplot(2, 3, k);
stem(ts, y, 'filled');
grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title(['Task 3: F_s = ' num2str(fs_samp) ' Hz']);
end

%Task 4: Analysis of Sampling and Reconstruction Quality Using MSE and FFT
clc;
clear all;
close all;
A = 0.5;
fo = 3000;
fs_ideal = 1e5;
n = 4;
to = 1/fo;
dur = n*to;
t = 0:1/fs_ideal:dur;
x = A*cos(2*pi*fo*t);
fs_list = [1e5 1e4 6e3 12e3 4e3 5e3];
MSEs = zeros(size(fs_list));
for k = 1:numel(fs_list)
fs_samp = fs_list(k);
ts = 0:1/fs_samp:dur;
y = A*cos(2*pi*fo*ts);
%a = length(t);
%b = length(ts);
%xq = 0: a/b : 5*fs_list(k)/fo;
z = interp1(ts, y, t, 'spline');
MSEs(k) = mean((x - z).^2);
fprintf('fs = %5d Hz | MSE (linear) = %.6g\n', fs_list, MSEs(k));
figure(1);
subplot(2,3,k);
stem(ts, y);
xlabel('Time (s)');
ylabel('Amplitude');
title(['fs = ' num2str(fs_samp) ' Hz']);
grid on;
figure(2);
subplot(2,3,k);
stem(z);
xlabel('Time (s)');
ylabel('Amplitude');
title(['fs = ' num2str(fs_samp) ' Hz']);
grid on;
N = length(y);
f = (-N/2:N/2-1)*(fs_samp/N);
Y = fft(y,64);
figure(3);
subplot(2,3,k);
plot(abs(Y));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title(['fs = ' num2str(fs_samp) ' Hz']);
grid on;
Z = fft(z,64);
figure(4);
subplot(2,3,k);
plot(abs(Z));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title(['Reconstructed fs = ' num2str(fs_samp) ' Hz']);
grid on;
end
figure;
stem(MSEs);
grid on;
xlabel('Sampling rate Fs (kHz)');
ylabel('MSE');
title('MSE vs Fs (linear interpolation)');
