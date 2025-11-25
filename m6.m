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
