%%bpsk_wireless

clc;
clear all;
close all;
nsym = 1e6;
x = randi(2,1,nsym)-1;
xbpsk = 2*x-1;
h = 1/sqrt(2)*(randn(1,nsym) + 1j*randn(1,nsym));
noise = 1/sqrt(1)*(randn(1,nsym) + 1j*randn(1,nsym));
snrdb = 0:3:30;
snrlin = 10.^(snrdb./10);
nsd = 1./sqrt(snrlin);
for i = 1:length(snrdb)
    y = h.*xbpsk + noise.*nsd(i);
    yeq = y./h;
    detect = real(yeq) > 0;
    %ber_pract(i) = (sum(abs(x-detect).^2))/nsym
    ber_pract(i) = sum(x ~= detect) / nsym;
end
ber_theo = 0.5*(1-sqrt(snrlin./(2+snrlin)))
semilogy(snrdb,ber_pract,'-+');
grid on;
hold on;
semilogy(snrdb,ber_pract,'-o');
grid on;

###
QPSK_WIRELESS
clc;
clear all;
close all;
nsym = 1e6;
xreal = randi(2,1,nsym)-1;
ximg = randi(2,1,nsym)-1;
xreal_bpsk = 2*xreal-1;
ximg_bpsk = 2*ximg-1;
xqpsk = (1/sqrt(2))*(xreal_bpsk + 1j*ximg_bpsk);
h = (1/sqrt(2))(randn(1,nsym) +1j randn(1,nsym));
noise = (1/sqrt(2))*(randn(1,nsym) + 1j*randn(1,nsym));
snrdb = 0:3:30;
snrlin = 10.^(snrdb./10);
nsd = 1./sqrt(snrlin);
for i = 1:length(snrdb)
     y = h.*xqpsk + noise.*nsd(i);
     yeq = y./h;
     yeq_real = real(yeq) >0;
     yeq_img = imag(yeq)>0;
    % yreal_bpsk = 2*yeq_real-1;
    % yimg_bpsk = 2*yeq_img-1;
    % yqpsk = (1/sqrt(2))*(yreal_bpsk +1j*yimg_bpsk);
     error1 = (sum(abs(xreal - yeq_real ).^2));
     error2 = (sum(abs(ximg - yeq_img ).^2));
     ber_pract(i) = (error1+error2)/(2*nsym); 
end
ber_theo = 0.5*(1-sqrt(snrlin./(2+snrlin)))
semilogy(snrdb,ber_pract,'-+');
grid on;
hold on;
semilogy(snrdb,ber_theo,'-o');
grid on;
legend('simulated','theoritical');