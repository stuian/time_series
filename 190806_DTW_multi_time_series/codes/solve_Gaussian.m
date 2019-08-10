function t = solve_Gaussian()

iSuccess=load('iSuccess0_1.txt');
dSuccess=load('dSuccess0_1.txt');

meani=mean(iSuccess(1:end,10));
meand=mean(dSuccess(1:end,10));
vari=var(iSuccess(1:end,10));
vard=var(dSuccess(1:end,10));
syms threshold
solve((1/sqrt(2*pi*vari))*(vpa(sym('exp(1)'),50))^(-((threshold-meani)^2)/(2*vari))-(1/sqrt(2*pi*vard))*(vpa(sym('exp(1)'),50))^(-((threshold-meand)^2)/(2*vard)))
end