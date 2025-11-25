%Task 1: Manual DFT Computation for N = 2
clc;
clear all;
close all;
x = [1, 2];
L = 2;
N = L;
X = zeros(1, N);
for k = 0:N-1
X(k+1) = 0;
for n = 0:N-1
X(k+1) = X(k+1) + x(n+1) * exp((-1j * 2 * pi / N) * n * k);
end
end
X_builtin1 = fft(x, 2);
disp('Manual FFT:');
disp(X);
disp('Built-in FFT:');
disp(X_builtin1);
disp('Difference:');
disp(X - X_builtin1);

% Task 2: Manual DFT Computation for N = 1024 and Comparison
clc;
clear all;
close all;
x = rand(1, 1024);
N = length(x);
X_manual = zeros(1, N);
for k = 0:N-1
X_manual(k+1) = 0;
for n = 0:N-1
X_manual(k+1) = X_manual(k+1) + ...
x(n+1) * exp((-1j * 2 * pi / N) * n * k);
end
end
X_builtin = fft(x);
disp(['Signal length: ', num2str(N)]);
disp('First 5 points of Manual FFT:');
disp(X_manual(1:5));
disp('First 5 points of Built-in FFT:');
disp(X_builtin(1:5));


%Task 3: Radix-2 FFT Implementation for N = 8
clc;
clear all;
close all;
x = [0,1,2,3,4,5,6,7];
L=length(x);
N=8;
if L==N
x1=x;
else
x1=[x, zeros(1, N-L)];
end
n=1:N;
bit=bitrevorder(n);
x=x1(bit);
level=log2(N);
X=x;
for i=1:level
m=2^i;
m2=m/2;
W_n=exp(-1j*2*pi/m);
for k=1:m:N
for j=0:m2-1
F1=X(k+j);
F2=(W_n^j) * X(k+j+m2);
X(k+j)=F1+F2;
X(k+j+m2)=F1-F2;
end
end
end
disp('Manual FFT:');
disp(X);
X_builtin1 = fft(x1, 8);
disp('Built-in FFT:');
disp(X_builtin1);

%Task 4: Complexity Comparison Between DFT and FFT
close all;
clear all;
clc;
N = [2 4 8 16 128 256 512 1024 2048];
for i=1:length(N)
speed(i) = N(i)^2/((N(i)/2)*log2(N(i)));
end
plot(N,speed,"-x");
grid on;
title("Time complexity vs N");
xlabel("N");
ylabel("Time complexity");

%Task 5: Radix-2 FFT Implementation for N = 4
clc;
clear all;
close all;
x = [0,1,2,3,4,5,6,7];
L=length(x);
N=4; % Changed from 8 to 4
if L==N
x1=x;
else
x1=[x, zeros(1, N-L)];
end
n=1:N;
bit=bitrevorder(n);
x=x1(bit);
level=log2(N);
X=x;
for i=1:level
m=2^i;
m2=m/2;
W_n=exp(-1j*2*pi/m);
for k=1:m:N
for j=0:m2-1
F1=X(k+j);
F2=(W_n^j) * X(k+j+m2);
X(k+j)=F1+F2;
X(k+j+m2)=F1-F2;
end
end
end
disp('Manual FFT:');
disp(X);
X_builtin1 = fft(x1, 4); % Changed from 8 to 4
disp('Built-in FFT:');
disp(X_builtin1);
