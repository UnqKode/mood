clc;
clear all;
close all;

Nsym = 1000;
x1 = randi(2, 1, Nsym) - 1;
x = 2 * x1 - 1;

h1 = (1 / sqrt(2)) * complex(randn(1, Nsym), randn(1, Nsym));
h2 = (1 / sqrt(2)) * complex(randn(1, Nsym), randn(1, Nsym));

n1 = (1 / sqrt(2)) * complex(randn(1, Nsym), randn(1, Nsym));
n2 = (1 / sqrt(2)) * complex(randn(1, Nsym), randn(1, Nsym));

y1 = h1 .* x + n1;
y2 = h2 .* x + n2;

zdetect = (conj(h1) .* y1+conj(h2) .* y2) ./ (conj(h1) .* h1 + conj(h2) .* h2);

det = 2 * (zdetect > 0) - 1;

error = sum(abs(det ~= x)) / Nsym;

disp(error);