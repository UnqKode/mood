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
