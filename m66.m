%%OSTBC(Alamouti Code)%%
clc;
clear all;
close all;
snrdb=0:3:30;
snrlin=10.^(snrdb./10);
n=1e6;
xts1=randi(2,1,n)-1;
xts2=randi(2,1,n)-1;
x1=2*xts1-1; 
x2=2*xts2-1;    
h1=1/sqrt(2)*(complex(randn(1,n),randn(1,n)));
h2=1/sqrt(2)*(complex(randn(1,n),randn(1,n)));
for i=1:length(snrdb)
n1=1/sqrt(snrlin(i))*(complex(randn(1,n),randn(1,n)));
n2=1/sqrt(snrlin(i))*(complex(randn(1,n),randn(1,n)));
yts1=h1.*x1+h2.*x2+n1;
yts2=h1.*(-conj(x2))+h2.*conj(x1)+n2;
yts2c=conj(yts2);
nh=sqrt(h1.*conj(h1)+conj(h2).*h2);
c1=[h1;conj(h2)];
y=[yts1;yts2c];
v1=c1./nh;
v1h=[conj(h1);h2]./nh;
z1=v1h.*y;
z2=sum(z1);
z3=z2>0;
error(i)=sum(abs(z3~=xts1)/n);  
end
l=2;
theo = factorial(2*l-1)/(factorial(l-1)*factorial(l))*(snrlin).^(-l);
semilogy(snrdb,theo,"o-");
hold on;
theo2 = factorial(2*l-1)/(factorial(l-1)*factorial(l))*(2*snrlin).^(-l);
semilogy(snrdb,theo2,"g");
hold on ;
semilogy(snrdb,error,"+-");
grid on ;
legend("theo","theo2", "error")
