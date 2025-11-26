%%MRT%%
clc;
clear all;
close all;

N = 1e6;
snr_db  = 0:3:30;
snr_lin = 10.^(snr_db/10);

L = 3;

for i = 1:length(snr_db)
    x1 = randi([0,1],1,N);
    x  = 2*x1 - 1;

    h_store = cell(1,L);
    for Tx_ant = 1:L
        h_store{Tx_ant} = (randn(1,N) + 1j*randn(1,N))/sqrt(2); 
    end

    norm = zeros(1,N);
    for Tx_ant = 1:L
        h = h_store{Tx_ant};
        norm = norm + abs(h).^2;
    end

    y1 = zeros(1,N);
    for Tx_ant = 1:L
        h  = h_store{Tx_ant};
        w  = conj(h)./sqrt(norm);
        tx = w.*x;              
        y1 = y1 + h.*tx;         
    end

    n = (randn(1,N) + 1j*randn(1,N))/sqrt(1);
    y = y1 + n/sqrt(snr_lin(i));
    z = y ./ sqrt(norm);
    detect = real(z)>0;

    error(i)   = sum(abs(x1 - detect).^2);
    ber_sim(i) = error(i)/N;
    ber_theoritical(i) = ((factorial(2*L-1))/((factorial(L))*(factorial(L-1)))).*((1./(2*snr_lin(i))).^L);
end

semilogy(snr_db,ber_sim,'-+'); hold on; grid on;
semilogy(snr_db,ber_theoritical,'-o');
xlabel('snr\_dB'); ylabel('BER');
title('BPSK FOR N TX ANTENNA');
legend('simulated','theoritical');

for i = 1:length(snr_db)
    fprintf('%3d dB , %10d , %.3e \n', snr_db(i), error(i), ber_sim(i));
end
