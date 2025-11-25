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
