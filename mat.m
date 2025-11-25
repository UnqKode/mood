%%%%EXP-1%%%%
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


%%%%EXP-2%%%%
%Task 1: Sampling at 8 kHz
clc;
clear all;
close all;
A = 0.5;
f = 3000;
fs = 8000;
num_cycles = 4;
Ts = 1/fs;
samples_per_cycle = fs / f;
total_samples = ceil(num_cycles * samples_per_cycle);
n = 0:total_samples-1;
t = n * Ts;
y_n = A * cos(2 * pi * f * n * Ts);
stem(t, y_n, 'filled');
title('Continuous Signal with Sampling Points');
xlabel('Time (s)');
ylabel('Amplitude (V)');

%Task  2: Quantizer function
function[y]= myquantizer(x,level)
delta=(max(x)-min(x))/(level-1);
a=x-min(x);
b=a/delta;
c=round(b);
d=c*delta+min(x);
y=d;
end

clc;
close all;
clear all;
a = 1;
f = 3e3;
fs = 8e3;
ts = 1/fs;
n = 0:1:(5*fs/f); % sample index
x = a*sin(2*pi*f*n*ts);
L = [8,16,32,64,128];
mse = zeros(1,length(L));
for i = 1:length(L)
y = myquantizer(x, L(i));
subplot(2,3,i);
stem(n, y);
title(['L = ', num2str(L(i))]);
mse(i) = mean((x - y).^2);
end
subplot(2,3,6);
stem(L, mse);
title('MSE vs Levels');
xlabel('Levels');
ylabel('MSE');

%Task 3: Efect of Quantization Levels on Sampled Waveform
clc;
clear all;
close all;
A = 0.5;
f = 3000;
fs = 8000;
num_cycles = 4;
Ts = 1/fs;
n = 0:ceil(num_cycles * fs / f);
t = n * Ts;
y = A * cos(2 * pi * f * n * Ts);
L_levels = [8, 16, 32, 64];
subplot(5, 1, 1);
stem(t, y, 'filled');
title('Original Sampled Signal y(t)');
17
EXPERIMENT 2. DISCRETIZATION OF SIGNALS : QUANTIZATION AND ENCODING
xlabel('Time (s)');
ylabel('Amplitude (V)');
grid on;
for i = 1:length(L_levels)
L = L_levels(i);
g = myquantizer(y, L);
subplot(5, 1, i+1);
stem(t, g, 'filled');
title(sprintf('Quantized Signal with L = %d levels', L));
xlabel('Time (s)');
ylabel('Amplitude (V)');
grid on;
end

%Task 4: MSE vs SQNR vs Quantization
clc;
close all;
clear all;
A = 0.5;
f = 3000;
fs = 8000;
num_cycles = 4;
Ts = 1/fs;
n = 0:ceil(num_cycles * fs / f);
t = n * Ts;
x = A * cos(2 * pi * f * n * Ts);
L = [8, 16, 32, 64];
b = log2(L);
mse = zeros(size(L));
sqnr_measured = zeros(size(L));
sqnr_theoretical = zeros(size(L));
for i = 1:length(L)
y = myquantizer(x, L(i));
% (MSE)
mse(i) = mean((x - y).^2);
% SQNR
signal_power = mean(x.^2);
noise_power = mse(i);
sqnr_measured(i) = 10 * log10(signal_power / noise_power);
% Compute theoretical
sqnr_theoretical(i) = 1.76 + 6.02 * b(i);
% Plot
subplot(2, 3, i);
stem(t, x);
hold on;
stem(t, y);
title(sprintf('L = %d (b = %d bits)', L(i), b(i)));
xlabel('Time (s)');
ylabel('Amplitude (V)');
grid on;
legend('Original', 'Quantized');
hold off;
end
% Plot MSE results
subplot(2, 3, 5);
stem(L, mse);
title('(MSE) vs Quantization Levels');
xlabel('Number of Levels (L)');
ylabel('MSE (V^2)');
grid on;
% Plot SQNR comparison
subplot(2, 3, 6);
stem(b, sqnr_measured);
hold on;
stem(b, sqnr_theoretical);
title('SQNR Comparison');
xlabel('Number of Bits (b)');
ylabel('SQNR (dB)');
grid on;
legend('Measured SQNR', 'Theoretical SQNR');

%Task 5: SQNR vs Input ampli
clc;
clear all;
close all;
A = 1:1:10;
level = 64;
f = 3*10^3;
fs = 8*10^3;
ts = 1/fs;
n = 0:5*fs/f;
for i = 1:length(A)
x = A(i)*cos(2*pi*f*n*ts);
z = myquantizer(x, level);
error(i) = mean(((x - z).^2));
end
SQNR = A./error;
SQNRdb = 20*log(SQNR);
subplot(2,1,1);
stem(A, SQNRdb);
title(" SQNRdb vs input voltage");
xlabel("A (input voltage)");
ylabel("SQNRdb");
subplot(2,1,2);
stem(A, SQNR);
title(" SQNR vs input voltage");
xlabel("A (input voltage)");

%Task 6: Encoding
clc;
clear all;
close all;
A = 1;
f = 3000;
fs = 9000;
ts = 1/fs;
n = 0:5*fs/f;
x = A*cos(2*pi*f*n*ts);
level = 8;
y = zeros(length(x),3);
z = myquantizer(x, level);
for i = 1:length(z)
if (z(i) >= -1 && z(i) < -0.72)
y(i,:) = [0,0,0];
end
if (z(i) >= -0.72 && z(i) < -0.44)
y(i,:) = [0,0,1];
end
if (z(i) >= -0.44 && z(i) < -0.16)
y(i,:) = [0,1,0];
end
if (z(i) >= -0.16 && z(i) < 0.12)
y(i,:) = [0,1,1];
end
if (z(i) >= 0.12 && z(i) < 0.4)
y(i,:) = [1,0,0];
end
if (z(i) >= 0.4 && z(i) < 0.68)
y(i,:) = [1,0,1];
end
if (z(i) >= 0.68 && z(i) <= 1)
y(i,:) = [1,1,1];
end
end
t = (n+1)*ts;
decimal_values = y(:,1)*4 + y(:,2)*2 + y(:,3)*1;
figure;
subplot(3,1,1);
stem(x);
title("sampled Signal");
subplot(3,1,2);
stem(z);
title("quantized Signal");
subplot(3,1,3);
stem(t, decimal_values);
title('Encoded Signal');
xlabel('Time');
ylabel('Value');
grid on;

%%%EXP-3%%%
%Task 1: Linear Convolution Using Convolution Matrix
clc;
clear all;
close all;
x = [0, 1, 2, 3];
h = [1, 1];
n_x = 0:length(x)-1;
n_h = 0:length(h)-1;
m = length(x);
o = length(h);
l = m + o - 1;
M = zeros(l, m);
Y = zeros(1, l);
for i = 1:l
for j = 1:m
if (i - j + 1 > 0) && (i - j + 1 <= o)
Y(i) = Y(i) + x(j) * h(i - j + 1);
M(i, j) = h(i - j + 1);
else
M(i, j) = 0;
end
end
end
disp('Convolution matrix M:');
disp(M);
disp('Convolution result Y:');
disp(Y);
inbuilt_y = conv(x, h);
disp('Built-in conv() result:');
disp(inbuilt_y);
% Plotting
figure;
subplot(2, 2, 1);
stem(n_x, x, 'b', 'filled');
title('Input signal x[n] = [0,1,2,3]');
xlabel('n'); ylabel('x[n]');
grid on;
subplot(2, 2, 2);
stem(n_h, h, 'r', 'filled');
title('Impulse response h[n] = [1,1]');
xlabel('n'); ylabel('h[n]');
grid on;
subplot(2, 2, 3);
stem(0:length(Y)-1, Y, 'g', 'filled');
title('PRACTICAL OUTPUT (Matrix conv)');
xlabel('n'); ylabel('y[n]');
grid on;
subplot(2, 2, 4);
stem(0:length(inbuilt_y)-1, inbuilt_y, 'm', 'filled');
title('INBUILT CONVOLUTION');
xlabel('n'); ylabel('y[n]');
grid on;
% Verify results match
max_error = max(abs(Y - inbuilt_y));
fprintf('Maximum error between implementations: %.2e\n', max_error);
% Show that matrix multiplication gives same result
fprintf('Verification using matrix multiplication:\n');
disp('M * x'' =');
disp(M * x');
disp('Should equal Y'' =');
disp(Y');

%Task 2: Linear Convolution for Different Inputs and Impulse Responses
clc;
clear all;
close all;
h1 = [-1, 1];
h2 = [-1, -1];
h3 = [1/3, -1/3, 1/3];
h4 = [1/4, 1/4, -1, 1/4, 1/4];
h_list = {h1, h2, h3, h4};
h_names = {'h1(n) = [-1,1]', 'h2(n) = [-1,-1]', ...
'h3(n) = [1/3,-1/3,1/3]', 'h4(n) = [1/4,1/4,-1,1/4,1/4]'};
frequencies = [0, 1/10, 1/5, 1/4, 1/2];
freq_names = {'0', '1/10', '1/5', '1/4', '1/2'};
n = 0:99; % 100 samples
for h_idx = 1:length(h_list)
h = h_list{h_idx};
h_name = h_names{h_idx};
for f_idx = 1:length(frequencies)
f = frequencies(f_idx);
freq_name = freq_names{f_idx};
x = cos(2*pi*f*n);
y_custom = myLinConvMat(x, h);
inbuilt_y = conv(x, h);
figure;
subplot(3,1,1);
stem(n, x);
title(sprintf('Input: cos(2*%s*n) with %s', freq_name, h_name));
xlabel('n');
ylabel('Amplitude');
grid on;
subplot(3,1,2);
stem(0:length(y_custom)-1, y_custom);
title('Output using myLinConvMat');
xlabel('n');
ylabel('Amplitude');
grid on;
subplot(3,1,3);
stem(0:length(inbuilt_y)-1, inbuilt_y);
title('Output using built-in conv');
xlabel('n');
ylabel('Amplitude');
grid on;
end
end

%Task 3: DFT of Impulse Responses Using myDft and fft
clc;
clear all;
close all;
% Define impulse responses from Exercise 1
h1 = [-1, 1];
h2 = [-1, -1];
h3 = [1/3, -1/3, 1/3];
h4 = [1/4, 1/4, -1, 1/4, 1/4];
h_list = {h1, h2, h3, h4};
h_names = {'h1(n)', 'h2(n)', 'h3(n)', 'h4(n)'};
% DFT sizes to test
N_points = [8, 16, 32, 64];
% Test each impulse response
for h_idx = 1:length(h_list)
h = h_list{h_idx};
h_name = h_names{h_idx};
fprintf('\n=== %s ===\n', h_name);
% Test each DFT size
for n_idx = 1:length(N_points)
N = N_points(n_idx);
% Compute DFT using custom function
X_myDft = myDft(h, N);
% Compute DFT using MATLAB's fft (with zero-padding if needed)
if length(h) < N
h_padded = [h, zeros(1, N - length(h))];
X_fft = fft(h_padded);
else
X_fft = fft(h, N);
end
% Calculate maximum difference
max_diff = max(abs(X_myDft - X_fft));
fprintf('N = %2d: Max difference = %.4e\n', N, max_diff);
% Plot only magnitude comparison
figure;
stem(0:N-1, abs(X_myDft), 'r', 'filled', ...
'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
stem(0:N-1, abs(X_fft), 'bo', ...
'LineWidth', 1, 'MarkerSize', 4);
title(sprintf('%s - Magnitude Comparison (N=%d)', h_name, N));
xlabel('Frequency index k');
ylabel('|X(k)|');
legend('myDft', 'MATLAB fft');
grid on;
end
end

%Task 4:Task 4: DFT Matrix Generation and Verification
clc;
clear all;
close all;
% Define impulse responses from Exercise 1
h1 = [-1, 1];
h2 = [-1, -1];
h3 = [1/3, -1/3, 1/3];
h4 = [1/4, 1/4, -1, 1/4, 1/4];
h_list = {h1, h2, h3, h4};
h_names = {'h1(n)', 'h2(n)', 'h3(n)', 'h4(n)'};
% DFT sizes to test (using smaller sizes for matrix display)
N_points = [4, 8]; % Smaller sizes for readable matrix display
fprintf('DFT Matrix and Comparison: myDft vs MATLAB fft\n');
fprintf('==============================================\n\n');
for h_idx = 1:length(h_list)
h = h_list{h_idx};
h_name = h_names{h_idx};
fprintf('\n=== %s = [', h_name);
fprintf('%g ', h);
fprintf('] ===\n\n');
% Test each DFT size
for n_idx = 1:length(N_points)
N = N_points(n_idx);
% Get DFT matrix
W = mydftmatrix(N);
% Display full DFT matrix
fprintf('DFT Matrix for N = %d:\n', N);
for i = 1:N
for j = 1:N
real_part = real(W(i,j));
imag_part = imag(W(i,j));
if imag_part >= 0
fprintf('%.2f+%.2fi ', real_part, imag_part);
else
fprintf('%.2f%.2fi ', real_part, imag_part);
end
end
fprintf('\n');
end
fprintf('\n');
% Compute DFT using matrix multiplication
if length(h) < N
h_padded = [h, zeros(1, N - length(h))];
X_myDft = (W * h_padded.').';
else
X_myDft = (W * h(1:N).').';
end
% Compute DFT using MATLAB's fft
if length(h) < N
h_padded = [h, zeros(1, N - length(h))];
X_fft = fft(h_padded);
else
X_fft = fft(h, N);
end
% Display results
fprintf('Input sequence (zero-padded): [');
if length(h) < N
fprintf('%g ', h_padded);
else
fprintf('%g ', h(1:N));
end
fprintf(']\n');
fprintf('myDft result: [');
for k = 1:N
real_part = real(X_myDft(k));
imag_part = imag(X_myDft(k));
if imag_part >= 0
fprintf('%.4f+%.4fi ', real_part, imag_part);
else
fprintf('%.4f%.4fi ', real_part, imag_part);
end
end
fprintf(']\n');
fprintf('fft result: [');
for k = 1:N
real_part = real(X_fft(k));
imag_part = imag(X_fft(k));
if imag_part >= 0
fprintf('%.4f+%.4fi ', real_part, imag_part);
else
fprintf('%.4f%.4fi ', real_part, imag_part);
end
end
fprintf(']\n');
% Calculate maximum difference
max_diff = max(abs(X_myDft - X_fft));
fprintf('Max difference = %.4e\n\n', max_diff);
fprintf('----------------------------------------\n\n');
end
end

Function: myDft
function X = myDft(x, N)
% MYDFT Computes Discrete Fourier Transform
% X = myDft(x, N) returns N-point DFT of sequence x
% If input length is less than N, zero-pad
if length(x) < N
x = [x, zeros(1, N - length(x))];
end
% Initialize DFT output
X = zeros(1, N);
% Compute DFT using the formula
for k = 0:N-1
for n = 0:N-1
X(k+1) = X(k+1) + x(n+1) * exp(-1i * 2 * pi * k * n / N);
end
end
end

Function: mydftmatrix
function [y] = mydftmatrix(n)
y = zeros(n,n);
for i = 1:n
for j = 1:n
y(i,j) = exp(-1i*2*3.14*(i-1)*(j-1)/n);
end
end
end

Function: myLinConvMat
function Y = myLinConvMat(x, h)
% MYLINCONVMAT Custom convolution function
% Y = myLinConvMat(x, h) performs convolution of x and h
m = length(x);
o = length(h);
l = m + o - 1;
X = [x, zeros(1, o)];
H = [h, zeros(1, m)];
Y = zeros(1, l);
for i = 1:l
for j = 1:m
if (i - j + 1 > 0)
Y(i) = Y(i) + X(j) * H(i - j + 1);
end
end
end
end

%%%EXP-4%%%

%Task 1: Circular Convolution Using Circulant Matrix and cconv()
clc;
clear all;
close all;
x = [1, 2, 3, 4];
h = [2, 1, 2, 1];
[H, circular_conv] = myCircularConvMat(x, h);
N = max(length(x), length(h));
circular_conv_inbuilt = cconv(x, h, N);
% Display results
disp('Circulant Matrix H:');
disp(H);
disp('My circular convolution result:');
disp(circular_conv');
disp('Built-in cconv result:');
disp(circular_conv_inbuilt);
% Plot comparison
figure;
subplot(2,1,1);
stem(circular_conv_inbuilt, 'filled', 'LineWidth', 2);
title('Built-in cconv result');
xlabel('Sample index');
ylabel('Amplitude');
grid on;
subplot(2,1,2);
stem(circular_conv', 'filled', 'LineWidth', 2, 'Color', 'r');
title('My circular convolution result');
xlabel('Sample index');
ylabel('Amplitude');
grid on;

%Task 2: Circular Convolution Using DFT and IDFT
clc;
clear all;
close all;
x = [1, 2, 3, 4, 5, 6, 7, 8];
h = [2, 1, 2, 1, 2, 1, 2, 1];
N = length(x); % Should be 8
% Method 1: Direct circular convolution using circulant matrix
[H, circular_conv] = myCircularConvMat(x, h);
% Method 2: Circular convolution using DFT
H_dft = myDFT([], N); % DFT matrix
% Compute DFT of both signals
X_dft = H_dft * x'; % DFT of x
H_dft_values = H_dft * h'; % DFT of h
% Multiply in frequency domain (circular convolution property)
Y = H_dft_values .* X_dft;
% Compute inverse DFT
H_idft = myIDFT(N);
z = H_idft * Y; % No need for extra 1/N since it's already in myIDFT
% Compare results
disp('Direct circular convolution result:');
disp(circular_conv');
disp('DFT-based circular convolution result:');
disp(real(z')); % Should be very similar
disp('Difference:');
disp(max(abs(circular_conv' - real(z'))));
% Plot comparison
figure;
subplot(2,1,1);
stem(circular_conv, 'filled', 'LineWidth', 2);
title('Direct Circular Convolution');
xlabel('Sample index');
ylabel('Amplitude');
grid on;
subplot(2,1,2);
stem(real(z), 'filled', 'LineWidth', 2, 'Color', 'r');
title('DFT-based Circular Convolution');
xlabel('Sample index');
ylabel('Amplitude');
grid on;

%Task 3: Circular Convolution as Change of Basis with DFT Matrix
clc;
clear all;
close all;
x = [1, 2, 3, 4, 5, 6, 7, 8];
h = [2, 1, 2, 1, 2, 1, 2, 1];
N = length(x);
% a) Use myCircularConvMat to find convolution matrix H
[H, circular_conv] = myCircularConvMat(x, h);
% b) Use myDFT function to generate 8-point DFT matrix D8
D8 = myDFT([], N); % DFT matrix
% c) Find out 8-point DFT of sequence x(n) given by X_F(k)
X_F = D8 * x'; % DFT of x
% d) Calculate H_F = D8 · H · D8¹
% Note: Proper inverse from myIDFT
D8_inv = myIDFT(N); % This is the proper inverse
H_F = D8 * H * D8_inv;
% Check if H_F is diagonal
disp('Is H_F diagonal?');
is_diagonal = isdiag(H_F);
disp(is_diagonal);
if ~is_diagonal
disp('Maximum off-diagonal element:');
H_F_off_diag = H_F - diag(diag(H_F));
disp(max(abs(H_F_off_diag(:))));
end
% Compute Y_F = H_F · X_F
Y_F = H_F * X_F;
% Find y = D8¹ · Y_F
y = D8_inv * Y_F;
% e) Compare with circular convolution
disp('Circular convolution result (direct method):');
disp(circular_conv');
disp('Result from change of basis method:');
disp(real(y'));
disp('Difference between methods:');
difference = max(abs(circular_conv' - real(y')));
disp(difference);
% Plot comparison
figure;
subplot(2,1,1);
stem(circular_conv, 'filled', 'LineWidth', 2);
title('Direct Circular Convolution');
xlabel('Sample index');
ylabel('Amplitude');
grid on;
subplot(2,1,2);
stem(real(y), 'filled', 'LineWidth', 2, 'Color', 'r');
title('Change of Basis Method');
xlabel('Sample index');
ylabel('Amplitude');
grid on;

Function: myCircularConvMat
function [H, circular_conv] = myCircularConvMat(x, h)
N = max(length(x), length(h));
x_padded = [x, zeros(1, N-length(x))];
h_padded = [h, zeros(1, N-length(h))];
H = zeros(N, N);
% Build circulant matrix correctly
for n = 1:N
for i = 1:N
% Each ROW is a circular shift of h
z = mod((i - n), N) + 1;
if z <= 0
z = z + N;
end
H(n, i) = h_padded(z);
end
end
circular_conv = H * x_padded';
end

Function: myDFT
function H_dft = myDFT(x, N)
% MYDFT generates an N-point DFT matrix
H_dft = zeros(N, N);
for k = 1:N
for n = 1:N
H_dft(k, n) = exp(-1i * 2 * pi * (k-1) * (n-1) / N);
end
end
end

Function: myIDFT
function H_idft = myIDFT(N)
% MYIDFT generates an N-point IDFT matrix
H_idft = zeros(N, N);
for k = 1:N
for n = 1:N
H_idft(k, n) = (1/N) * exp(1i * 2 * pi * (k-1) * (n-1) / N);
end
end
end

%%%EXP-8%%%

%Task 1: Manual DFT Computation for N = 2
clc;
clear all;
close all;
x = [1, 2];
L = 2;
N = L;
X = zeros(1, N);
for k = 0:N-1
X(k+1) = 0;
for n = 0:N-1
X(k+1) = X(k+1) + x(n+1) * exp((-1j * 2 * pi / N) * n * k);
end
end
X_builtin1 = fft(x, 2);
disp('Manual FFT:');
disp(X);
disp('Built-in FFT:');
disp(X_builtin1);
disp('Difference:');
disp(X - X_builtin1);

% Task 2: Manual DFT Computation for N = 1024 and Comparison
clc;
clear all;
close all;
x = rand(1, 1024);
N = length(x);
X_manual = zeros(1, N);
for k = 0:N-1
X_manual(k+1) = 0;
for n = 0:N-1
X_manual(k+1) = X_manual(k+1) + ...
x(n+1) * exp((-1j * 2 * pi / N) * n * k);
end
end
X_builtin = fft(x);
disp(['Signal length: ', num2str(N)]);
disp('First 5 points of Manual FFT:');
disp(X_manual(1:5));
disp('First 5 points of Built-in FFT:');
disp(X_builtin(1:5));


%Task 3: Radix-2 FFT Implementation for N = 8
clc;
clear all;
close all;
x = [0,1,2,3,4,5,6,7];
L=length(x);
N=8;
if L==N
x1=x;
else
x1=[x, zeros(1, N-L)];
end
n=1:N;
bit=bitrevorder(n);
x=x1(bit);
level=log2(N);
X=x;
for i=1:level
m=2^i;
m2=m/2;
W_n=exp(-1j*2*pi/m);
for k=1:m:N
for j=0:m2-1
F1=X(k+j);
F2=(W_n^j) * X(k+j+m2);
X(k+j)=F1+F2;
X(k+j+m2)=F1-F2;
end
end
end
disp('Manual FFT:');
disp(X);
X_builtin1 = fft(x1, 8);
disp('Built-in FFT:');
disp(X_builtin1);

%Task 4: Complexity Comparison Between DFT and FFT
close all;
clear all;
clc;
N = [2 4 8 16 128 256 512 1024 2048];
for i=1:length(N)
speed(i) = N(i)^2/((N(i)/2)*log2(N(i)));
end
plot(N,speed,"-x");
grid on;
title("Time complexity vs N");
xlabel("N");
ylabel("Time complexity");

%Task 5: Radix-2 FFT Implementation for N = 4
clc;
clear all;
close all;
x = [0,1,2,3,4,5,6,7];
L=length(x);
N=4; % Changed from 8 to 4
if L==N
x1=x;
else
x1=[x, zeros(1, N-L)];
end
n=1:N;
bit=bitrevorder(n);
x=x1(bit);
level=log2(N);
X=x;
for i=1:level
m=2^i;
m2=m/2;
W_n=exp(-1j*2*pi/m);
for k=1:m:N
for j=0:m2-1
F1=X(k+j);
F2=(W_n^j) * X(k+j+m2);
X(k+j)=F1+F2;
X(k+j+m2)=F1-F2;
end
end
end
disp('Manual FFT:');
disp(X);
X_builtin1 = fft(x1, 4); % Changed from 8 to 4
disp('Built-in FFT:');
disp(X_builtin1);






%%%EXP-6%%%

%Task 1: DCT and IDCT of an Image
clc;
clear all;
close all;
img = imread('cameraman.tif');
img_double = im2double(img);
dct_coeff = dct2(img_double);
reconstructed_img = idct2(dct_coeff);
figure;
subplot(1,3,1);
imshow(img);
title('Original Image');
subplot(1,3,2);
imshow(dct_coeff);
title('DCT Coefficients');
subplot(1,3,3);
imshow(reconstructed_img);
title('Reconstructed Image (IDCT)');

%Task 2: Horizontal Frequency Truncation in DCT Domain
clc;
clear all;
close all;
x=imread('cameraman.tif');
x_dct=dct2(x)
[M,N]=size(x);
for i=1:M
for j=1:N
if j<=N/2
x_dct(i,j)=x_dct(i,j);
else
x_dct(i,j)=0;
end
end
end
x_r=idct2(x_dct);
subplot(1,3,1);
imshow(x);
subplot(1,3,2);
imshow(x_dct);
subplot(1,3,3);
imshow(uint8(x_r));

%Task 3: Vertical Frequency Truncation in DCT Domain
clc;
clear all;
close all;
x=imread('cameraman.tif');
x_dct=dct2(x);
[M,N]=size(x);
for i=1:M
for j=1:N
if i<=M/2
x_dct(i,j)=x_dct(i,j);
else
x_dct(i,j)=0;
end
end
end
x_r=idct2(x_dct);
subplot(1,3,1);
imshow(x);
subplot(1,3,2);
imshow(x_dct);
subplot(1,3,3);
imshow(uint8(x_r));

%Task 4: Using Custom 2D-DCT (my_dct2) and Truncation
clc;
clear all;
close all;
x=imread('cameraman.tif');
y=double(x);
[M,N]=size(y);
x_dct=my_dct2(y);
for i=1:M
for j=1:N
if(i<=M/2 && j<=N)
x_dct(i,j) = x_dct(i,j);
else
x_dct(i,j) = 0;
end
end
end
x_r=idct2(x_dct);
subplot(1,3,1);
imshow(x);
subplot(1,3,2);
imshow(x_dct);
subplot(1,3,3);
imshow(uint8(x_r));

%Task 5: Block-based Transform Coding and MSE vs Compression Ratio
clc;
clear all;
close all;
x = imread("cameraman.tif");
y = double(x);
block_size = 8;
[M, N] = size(y);
cases = [64, 32, 16, 8];
mse_values = zeros(1, length(cases));
compression_ratios = zeros(1, length(cases));
for c = 1:length(cases)
N2 = cases(c);
z = zeros(M, N);
z_out = zeros(M, N);
for i = 1:block_size:M
for j = 1:block_size:N
block = y(i:i+block_size-1, j:j+block_size-1);
z_block = my_dct2(block);
z_flat = z_block(:);
z_flat = [z_flat(1:N2); zeros(block_size^2-N2, 1)];
z_reshape = reshape(z_flat, [block_size, block_size]);
z(i:i+block_size-1, j:j+block_size-1) = z_reshape;
z_idct = idct2(z_reshape);
z_out(i:i+block_size-1, j:j+block_size-1) = z_idct;
end
end
mse_values(c) = mean((y(:) - z_out(:)).^2)
compression_ratios(c) = N2 / block_size^2;
figure(1);
subplot(2, 1, 1);
imshow(x);
title('Original');
73
EXPERIMENT 6. TRANSFORM BASED LOSSY COMPRESSION
subplot(2, length(cases), c + length(cases));
imshow(uint8(z_out));
title(['Reconstructed, N2 = ', num2str(N2)]);
end
figure(2);
stem(compression_ratios, mse_values);
xlabel('Compression Ratio');
ylabel('MSE');
title('MSE vs Compression Ratio');

%Task 6: Reconstructed Image for Fixed Number of Coecients
clc;
clear all;
close all;
x=imread('cameraman.tif');
y=double(x);
block_size=8;
n2=32;
[M,N]=size(y);
z_out=zeros(M,N);
for i=1:block_size:M
for j=1:block_size:N
block=y(i:i+block_size-1 , j:j+block_size-1);
z=my_dct2(block);
z_flat=z( : );
z_comp=[z_flat(1:n2);zeros((block_size)^2-n2,1)];
z_reshape=reshape(z_comp,[block_size,block_size]);
z_idct=idct2(z_reshape);
z_out(i:i+block_size-1 , j:j+block_size-1)=z_idct;
end
end
imshow(uint8(z_out));

User-defined Function: my_dct2
MATLAB Code
function [out] = my_dct2(x)
[M,N] = size(x);
alpha_k = [sqrt(1/M), ones(1, M-1)*sqrt(2/M)];
alpha_l = [sqrt(1/N), ones(1, N-1)*sqrt(2/N)];
out = zeros(M, N);
for k = 1:M
for l = 1:N
for i = 1:M
for j = 1:N
out(k, l) = out(k, l) + x(i, j) * ...
cos((pi*(2*(i-1)+1)*(k-1))/(2*M)) * ...
cos((pi*(2*(j-1)+1)*(l-1))/(2*N));
end
end
out(k, l) = alpha_k(k)*alpha_l(l)*out(k, l);
end
end
end

%%%EXP-7%%%

%Task 1: DCT and IDCT of Audio Signal Using Built-in Functions
clc;
clear all;
close all;
[x,fs]=audioread('handel.wav');
y=dct(x);
x_out=idct(y);
subplot(3,1,1);
plot(x);
title('Original Signal');
subplot(3,1,2);
plot(y);
title('DCT');
subplot(3,1,3);
plot(x_out);
title('Reconstructed Signal');
soundsc(x_out,fs);

%Task 2: Custom DCT and IDCT for Audio Signal
clc;
clear all;
close all;
[x, Fs] = audioread("handel.wav");
X = transpose(x);
X = X(1:10000);
x_dct = mydct(X);
x_r = myidct(x_dct);
subplot(3,1,1);
plot(X);
title('Original Signal');
subplot(3,1,2);
plot(x_dct);
title('DCT');
subplot(3,1,3);
plot(x_r);
title('Reconstructed Signal');
soundsc(x_r, Fs);
built_dct = dct(X);
diff_dct = max(abs(x_dct - built_dct))
built_r = idct(built_dct);
diff_r = max(abs(X - built_r))

%Task 3: Block-based Audio Compression Using DCT Thresholding
clc;
clear all;
close all;
[x, Fs] = audioread("handel.wav");
blocksize = 256;
N = length(x);
z_out = zeros(1, N);
Thresold1 = 0.09;
Thresold2 = -0.09;
for i = 1:blocksize:N
end_i = min(i+blocksize-1, N);
block = x(i:end_i);
curr_size = length(block);
block_padded = block;
if curr_size < blocksize
block_padded(curr_size+1:blocksize) = 0;
end
z_block = mydct(block_padded);
z_block(z_block > Thresold2 & z_block < Thresold1) = 0;
z_idct = myidct(z_block);
z_out(i:end_i) = z_idct(1:curr_size);
end
subplot(2,1,1);
plot(x);
title('Original Signal');
subplot(2,1,2);
plot(z_out);
title('Reconstructed Signal');
soundsc(z_out,Fs);
audiowrite('outputcompression.wav', z_out', Fs);
x_trans=transpose(x);
error=immse(z_out,x_trans);
numZeros = numel(x_trans) - nnz(x_trans);
comp_ratio=(N-numZeros)/N;
display(error);
display(comp_ratio);

%Task 4: Effect of Threshold on Compression Performance
clc;
clear all;
close all;
[x, Fs] = audioread("handel.wav");
blocksize = 256;
N = length(x);
cases = [0.09, 0.5, 0.01];
mse_values = zeros(1, length(cases));
compression_ratios = zeros(1, length(cases));
for c = 1:length(cases)
Thresold1 = cases(c);
Thresold2 = -cases(c);
z_out = zeros(1, N);
kept = 0;
for i = 1:blocksize:N-256
block = x(i:i+blocksize-1);
z_block = mydct(block);
mask = (z_block > Thresold2 & z_block < Thresold1) == 0;
z_block(~mask) = 0;
kept = kept + sum(mask);
z_idct = myidct(z_block);
z_out(i:i+blocksize-1) = z_idct;
end
mse_values(c) = immse(z_out,transpose(x));
compression_ratios(c) = kept / N;
figure(1);
subplot(2, 1, 1);
plot(x);
title('Original Signal');
subplot(2, length(cases), c + length(cases));
plot(z_out);
title(['Reconstructed, Th = ', num2str(cases(c))]);
end
for c = 1:length(cases)
fprintf('Case %d (Th = %.2f): MSE = %.4f, Compression Ratio = %.3f\n', ...
c, cases(c), mse_values(c), compression_ratios(c));
end
figure(2);
stem(compression_ratios, mse_values);
xlabel('Compression Ratio');
xlabel('MSE');
title('MSE vs. Compression Ratio');

%User-defined Functions
Function: mydct
function [y] = mydct(x)
N = length(x);
alpha_k = [sqrt(1/N), ones(1, N-1) * sqrt(2/N)];
y = zeros(1, N);
for k = 1:N
for i = 1:N
y(k) = y(k) + x(i) * cos(pi * (2*(i-1)+1) * (k-1) / (2*N));
end
y(k) = alpha_k(k) * y(k);
end
Function: myidct
function [y] = mydct(x)
N = length(x);
alpha_k = [sqrt(1/N), ones(1, N-1) * sqrt(2/N)];
y = zeros(1, N);
for k = 1:N
for i = 1:N
y(k) = y(k) + x(i) * cos(pi * (2*(i-1)+1) * (k-1) / (2*N));
end
y(k) = alpha_k(k) * y(k);
end

%%%EXP-9%%%

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

