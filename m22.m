%%BER_QPSK

clc;
clear all;
close all;
nsym = 1e6;
xreal = randi(2,1,nsym)-1;
xreal_bpsk = 2*xreal-1;
ximg = randi(2,1,nsym)-1;
ximg_bpsk = 2*ximg-1;
xqpsk = 1/sqrt(2)*(xreal_bpsk + 1j*ximg_bpsk);
noise  = 1/sqrt(2)*(randn(1,nsym) + 1j*randn(1,nsym));
snrdb = 0:2:24;
snrlin = 10.^(snrdb./10);
nsd = 1./sqrt(snrlin);
for i = 1:length(snrdb)
    y = xqpsk + noise*nsd(i);
    
    yreal = real(y) >0;
    yimg = imag(y) > 0;
    yreal_bpsk = 2*yreal-1;
    yimg_bpsk = 2*yimg-1;
    yqpsk = 1/sqrt(2).*(yreal_bpsk + 1j*yimg_bpsk);
    ber_pract(i) = (sum(abs(xqpsk-yqpsk).^2))/nsym
end
ber_theo = 2*qfunc(sqrt(snrlin))
semilogy(snrdb , ber_pract , '-+');
grid on;
hold on;
semilogy(snrdb , ber_theo , '-o');
grid on;
