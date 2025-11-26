%%Lab-1 BER BPSK
clc;
clear all;
close all;
Nsym = 1e6;
x = randi(2,1,Nsym)-1;
xbpsk =  2*x-1;
snrdb = 0:1:10;
snrlin = 10.^(snrdb./10);
noise = randn(1,Nsym);
n_var = 1./snrlin;
nsd = 1./sqrt(snrlin);
for i = 1:length(snrdb)
    y = xbpsk + noise*nsd(i);
    detect = y>0;
    ber_pract(i) = (sum(abs(x - detect).^2))/Nsym
end
ber_theo = qfunc(sqrt(snrlin))
semilogy(snrdb , ber_pract , '-+');
grid on;
hold on;
semilogy(snrdb , ber_theo , '-o');
grid on;
