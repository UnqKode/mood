%% Generalized Code MIMO ZF Detector %%
clc; close all;
clear all;
N = 1e5; 
snrdb = 0:2:20; 
snrlinear = 10.^(snrdb/10);
Nt = input("Enter the number of Tx Antennas:");
Nr = input("Enter the number of Rx Antennas:"); 
L = Nr-Nt+1;
xtx = zeros(Nt, N); 
for tx_ant = 1:Nt 
    xn_temp = randi(2, 1, N) - 1; 
    xtx(tx_ant, :) = 2 * xn_temp -1;
end
for ind = 1:length(snrdb)
    H = zeros(Nr, Nt, N); 
    for rx_ant = 1:Nr 
        for tx_ant = 1:Nt
            H(rx_ant, tx_ant, :) = (1/sqrt(2)) * (complex(randn(1, N), randn(1, N))); 
        end 
    end
    n = zeros(Nr, N); 
    for rx_ant = 1:Nr 
        n(rx_ant, :) = (1/sqrt(snrlinear(ind))) * (complex(randn(1, N), randn(1,N)));
    end
    y = zeros(Nr, N); 
    for i = 1:N 
        y(:, i) = squeeze(H(:, :, i)) * xtx(:, i) + n(:, i);
    end
    z2 = zeros(Nt, N); 
    for i = 1:length(xtx) 
        y_vec = y(:, i); 
        h = squeeze(H(:, :,i)); 
        h_inverse = inv(ctranspose(h) * h) * ctranspose(h);
        z1 = h_inverse * y_vec; 
        z = z1> 0;
        z2(:, i) = 2 * z - 1;
    end
    error = zeros(1, Nt); 
    for j = 1:Nt
        error(j) = sum(abs(z2(j, :) ~= xtx(j, :))) / N;
    end
    for j = 1:Nt
        eval(['error' num2str(j) '(ind) = error(j);']);
    end
    BER(ind) = mean(error);
end

BER_TH = nchoosek(2*L-1, L) .* (1./(2.*snrlinear)).^L;
figure;
for j = 1:Nt
    semilogy(snrdb, eval(['error' num2str(j)]), '-+'); 
    hold on;
end 
semilogy(snrdb, BER, '--'); 
hold on; 
semilogy(snrdb, BER_TH); 
grid on;
xlabel('snrdb'); 
ylabel('error'); 
legend_str = {}; 
for stream = 1:Nt
    legend_str{end+1} = ['simulated error' num2str(stream)];
end
legend_str{end+1} = 'simulated BER';
legend_str{end+1} = 'Theoretical BER';
legend(legend_str);
