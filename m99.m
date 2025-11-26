%% generalized  code for BER of OFDM system %%
clc; 
close all; 
clear all;

Nsym = 64;
nitr = 1e4;
l = input('ENTER THE VALUE OF L = ');
cp = input('ENTER THE VALUE OF Cp = ');
snrdb = 0:3:30;
snrlinear = 10.^(snrdb ./ 10);
error2 = zeros(1, length(snrdb));

for i = 1:length(snrdb)
    error = zeros(1, nitr);             
    for ind = 1:nitr
        x1 = randi(2, 1, Nsym) - 1;     
        x = 2*x1 - 1;                   

        h = (1/sqrt(2)) * complex(randn(1, l), randn(1, l));

        xif = ifft(x);                  
        xcp = [ xif(Nsym-cp+1:Nsym), xif ];  

        n = complex(randn(1, Nsym+cp), randn(1, Nsym+cp));
        noise_scale = 1./sqrt(snrlinear(i));
        
        y = filter(h, 1, xcp) + noise_scale .* n;

        y_nocp = y(cp+1:end);          
        Yf = fft(y_nocp);               
        Hf = fft(h, Nsym);              

        Zf = Yf ./ Hf;                  

        dec_bits = real(Zf) > 0;        

        error(ind) = sum(dec_bits ~= x1) / Nsym;
    end
    error2(i) = mean(error);
end

theoretical = 0.5 * (1 - sqrt(((l ./ Nsym) .* snrlinear) ./ (2 + (l ./ Nsym) .* snrlinear)));

semilogy(snrdb, error2, 'b-o');
hold on;
semilogy(snrdb, theoretical, 'r--');
grid on;
legend('Simulated', 'Theoretical');
xlabel('SNR (dB)');
ylabel('BER');

%%%SIMULINK CODE%%%
clc;
clear all;
close all;

snr_db = 0:3:30;
ber_sim = zeros(size(snr_db));
ber_theo = zeros(size(snr_db));
for i = 1:length(snr_db)
    SNRdB = snr_db(i);
    out = sim("exp9_simulink.slx");
    ber_sim(i) = out.ber_sim(end);
    ber_theo(i) = out.ber_theo(end);
end

semilogy(snr_db,ber_sim,'-+');
hold on;
grid on;
semilogy(snr_db,ber_theo,'-o');
title('BER Performance of Orthogonal Frequency Division Multiplexing (OFDM) system')
legend("Practical","Theoritical");
xlabel("SNRdB");
ylabel("Error");