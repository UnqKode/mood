%% Generalized code for MIMO LMMSE Detector %%
clc; 
clear all;
close all;
N = 1e5;
Nr = input('Enter the no. of Rx Antenna:'); 
Nt = input('Enter the no. of Tx Antenna:'); 
b1 = randi(2, Nt, N) - 1; 
b2 = randi(2, Nt, N) - 1; 
x = ((2*b1 - 1) + 1j*(2*b2 - 1)) / sqrt(2); 
snrdb = 0:2:20; 
snrlin = 10.^(snrdb./10); 
error = zeros(Nt, length(snrdb)); 
for j = 1:length(snrdb)
    H_real = randn(Nr, Nt, N);
    H_imag = randn(Nr, Nt, N); 
    H = (H_real + 1j*H_imag) / sqrt(2);
    n_real = randn(Nr, N); 
    n_imag = randn(Nr, N);
    n = (n_real +1j*n_imag) / sqrt(2);
    y = zeros(Nr, N); 
    for i = 1:N 
        y(:, i) = H(:, :, i) * x(:, i) + n(:, i) / sqrt(snrlin(j));
    end
    z_hat = zeros(Nt, N); 
    for i = 1:N 
        hmat = H(:, :, i); 
        yvec = y(:, i);
        sigma2 = 1 / snrlin(j); 
        hinv = (hmat' * hmat + sigma2 * eye(Nt)) \hmat';
        z1 = hinv * yvec;
        z_real = real(z1) > 0; 
        z_imag = imag(z1) > 0; 
        z_hat(:, i) =(2*z_real - 1) + 1j*(2*z_imag - 1); 
    end
    for k = 1:Nt 
        b1_hat = real(z_hat(k, :)) > 0; 
        b2_hat = imag(z_hat(k, :)) > 0; 
        error(k, j) = (sum(b1_hat ~=b1(k, :)) + sum(b2_hat ~= b2(k, :))) / (2 * N); 
    end
    snrdb(j); 
    for k = 1:Nt 
        fprintf('BER%d = %.4f', k, error(k,j)); 
        if k < Nt
            fprintf(', '); 
        else
            fprintf('\n');
        end
    end
end

markers = {'-+', '-*', '-o', '-s', '-d', '-^'}; 
for k = 1:Nt
    semilogy(snrdb, error(k, :), markers{k}); 
    hold on;
end 
grid on;
xlabel('SNR (dB)');
ylabel('Bit ErrorRate');
legend_entries = cell(1, Nt); 
for k =1:Nt 
    legend_entries{k} = ['Stream ', num2str(k), ' (LMMSE)'];
end 
legend(legend_entries); 
title(['MIMO LMMSE Detection Performance (', num2str(Nr), 'Rx, ', num2str(Nt),'Tx', 'QP']);
hold off;
