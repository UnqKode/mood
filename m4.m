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
