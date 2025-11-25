%Task 1: Sampling at 8 kHz
clc;
clear all;
close all;
A = 0.5;
f = 3000;
fs = 8000;
num_cycles = 4;
Ts = 1/fs;
samples_per_cycle = fs / f;
total_samples = ceil(num_cycles * samples_per_cycle);
n = 0:total_samples-1;
t = n * Ts;
y_n = A * cos(2 * pi * f * n * Ts);
stem(t, y_n, 'filled');
title('Continuous Signal with Sampling Points');
xlabel('Time (s)');
ylabel('Amplitude (V)');

%Task  2: Quantizer function
function[y]= myquantizer(x,level)
delta=(max(x)-min(x))/(level-1);
a=x-min(x);
b=a/delta;
c=round(b);
d=c*delta+min(x);
y=d;
end

clc;
close all;
clear all;
a = 1;
f = 3e3;
fs = 8e3;
ts = 1/fs;
n = 0:1:(5*fs/f); % sample index
x = a*sin(2*pi*f*n*ts);
L = [8,16,32,64,128];
mse = zeros(1,length(L));
for i = 1:length(L)
y = myquantizer(x, L(i));
subplot(2,3,i);
stem(n, y);
title(['L = ', num2str(L(i))]);
mse(i) = mean((x - y).^2);
end
subplot(2,3,6);
stem(L, mse);
title('MSE vs Levels');
xlabel('Levels');
ylabel('MSE');

%Task 3: Efect of Quantization Levels on Sampled Waveform
clc;
clear all;
close all;
A = 0.5;
f = 3000;
fs = 8000;
num_cycles = 4;
Ts = 1/fs;
n = 0:ceil(num_cycles * fs / f);
t = n * Ts;
y = A * cos(2 * pi * f * n * Ts);
L_levels = [8, 16, 32, 64];
subplot(5, 1, 1);
stem(t, y, 'filled');
title('Original Sampled Signal y(t)');
17
EXPERIMENT 2. DISCRETIZATION OF SIGNALS : QUANTIZATION AND ENCODING
xlabel('Time (s)');
ylabel('Amplitude (V)');
grid on;
for i = 1:length(L_levels)
L = L_levels(i);
g = myquantizer(y, L);
subplot(5, 1, i+1);
stem(t, g, 'filled');
title(sprintf('Quantized Signal with L = %d levels', L));
xlabel('Time (s)');
ylabel('Amplitude (V)');
grid on;
end

%Task 4: MSE vs SQNR vs Quantization
clc;
close all;
clear all;
A = 0.5;
f = 3000;
fs = 8000;
num_cycles = 4;
Ts = 1/fs;
n = 0:ceil(num_cycles * fs / f);
t = n * Ts;
x = A * cos(2 * pi * f * n * Ts);
L = [8, 16, 32, 64];
b = log2(L);
mse = zeros(size(L));
sqnr_measured = zeros(size(L));
sqnr_theoretical = zeros(size(L));
for i = 1:length(L)
y = myquantizer(x, L(i));
% (MSE)
mse(i) = mean((x - y).^2);
% SQNR
signal_power = mean(x.^2);
noise_power = mse(i);
sqnr_measured(i) = 10 * log10(signal_power / noise_power);
% Compute theoretical
sqnr_theoretical(i) = 1.76 + 6.02 * b(i);
% Plot
subplot(2, 3, i);
stem(t, x);
hold on;
stem(t, y);
title(sprintf('L = %d (b = %d bits)', L(i), b(i)));
xlabel('Time (s)');
ylabel('Amplitude (V)');
grid on;
legend('Original', 'Quantized');
hold off;
end
% Plot MSE results
subplot(2, 3, 5);
stem(L, mse);
title('(MSE) vs Quantization Levels');
xlabel('Number of Levels (L)');
ylabel('MSE (V^2)');
grid on;
% Plot SQNR comparison
subplot(2, 3, 6);
stem(b, sqnr_measured);
hold on;
stem(b, sqnr_theoretical);
title('SQNR Comparison');
xlabel('Number of Bits (b)');
ylabel('SQNR (dB)');
grid on;
legend('Measured SQNR', 'Theoretical SQNR');

%Task 5: SQNR vs Input ampli
clc;
clear all;
close all;
A = 1:1:10;
level = 64;
f = 3*10^3;
fs = 8*10^3;
ts = 1/fs;
n = 0:5*fs/f;
for i = 1:length(A)
x = A(i)*cos(2*pi*f*n*ts);
z = myquantizer(x, level);
error(i) = mean(((x - z).^2));
end
SQNR = A./error;
SQNRdb = 20*log(SQNR);
subplot(2,1,1);
stem(A, SQNRdb);
title(" SQNRdb vs input voltage");
xlabel("A (input voltage)");
ylabel("SQNRdb");
subplot(2,1,2);
stem(A, SQNR);
title(" SQNR vs input voltage");
xlabel("A (input voltage)");

%Task 6: Encoding
clc;
clear all;
close all;
A = 1;
f = 3000;
fs = 9000;
ts = 1/fs;
n = 0:5*fs/f;
x = A*cos(2*pi*f*n*ts);
level = 8;
y = zeros(length(x),3);
z = myquantizer(x, level);
for i = 1:length(z)
if (z(i) >= -1 && z(i) < -0.72)
y(i,:) = [0,0,0];
end
if (z(i) >= -0.72 && z(i) < -0.44)
y(i,:) = [0,0,1];
end
if (z(i) >= -0.44 && z(i) < -0.16)
y(i,:) = [0,1,0];
end
if (z(i) >= -0.16 && z(i) < 0.12)
y(i,:) = [0,1,1];
end
if (z(i) >= 0.12 && z(i) < 0.4)
y(i,:) = [1,0,0];
end
if (z(i) >= 0.4 && z(i) < 0.68)
y(i,:) = [1,0,1];
end
if (z(i) >= 0.68 && z(i) <= 1)
y(i,:) = [1,1,1];
end
end
t = (n+1)*ts;
decimal_values = y(:,1)*4 + y(:,2)*2 + y(:,3)*1;
figure;
subplot(3,1,1);
stem(x);
title("sampled Signal");
subplot(3,1,2);
stem(z);
title("quantized Signal");
subplot(3,1,3);
stem(t, decimal_values);
title('Encoded Signal');
xlabel('Time');
ylabel('Value');
grid on;
