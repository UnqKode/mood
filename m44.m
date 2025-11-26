%%MRC%%
clc;
clear all;
close all;

N = 1e6;
snr_db = 0:3:30;
snr_lin = 10.^(snr_db/10);

L = 3;

for i = 1:length(snr_db)
    x1 = randi([0,1],1,N);
    x = 2*x1 - 1;
    
    num = zeros(1,N);
    den = zeros(1,N);

    for Rx_ant = 1:L
        h = (randn(1,N) + 1j*randn(1,N))/sqrt(2);
        n = (randn(1,N) + 1j*randn(1,N))/sqrt(1);

        y = h.*x + n/sqrt(snr_lin(i));

        num = num + conj(h).*y;
        den = den + abs(h).^2;
    end

    z = num./den;
    detect = real(z)>0;
    error(i) = sum(abs(x1 - detect).^2);
    ber_sim(i) = error(i) / N;

    ber_theoritical(i) = ((factorial(2*L-1))/((factorial(L))*(factorial(L-1)))).*((1./(2*snr_lin(i))).^L);
end

semilogy(snr_db,ber_sim,'-+');
hold on;
grid on;
semilogy(snr_db,ber_theoritical,'-o');
xlabel('snr_db');
ylabel('ber');
title('BPSK FOR N RX ANTENNA');
legend('theroitical','simulated');

for i = 1:length(snr_db)
    fprintf('%3d dB , %10d , %.3e \n',snr_db(i),error(i),ber_sim(i));
end
