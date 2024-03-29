% 1. Señales de voz
load('sonidos2.mat');
% Grafica de las señales
figure("Name","Señal original");
subplot(3, 2, 1);
plot(a);
title('Vocal ''a''');
subplot(3, 2, 2);
plot(o);
title('Vocal ''o''');
subplot(3, 2, 3);
plot(p);
title('Letra ''p''');
subplot(3, 2, 4);
plot(s);
title('Letra ''s''');
subplot(3, 2, 5);
plot(pasa);
title('Palabra ''pasa''');
subplot(3, 2, 6);
plot(paso);
title('Palabra ''paso''');
hold on;

% 2. Filtro de preenfasis

preemphasis = [1, -0.95];
a_preemp = filter(preemphasis, 1, a);
o_preemp = filter(preemphasis, 1, o);
p_preemp = filter(preemphasis, 1, p);
s_preemp = filter(preemphasis, 1, s);
pasa_preemp = filter(preemphasis, 1, pasa);
paso_preemp = filter(preemphasis, 1, paso);
figure;
freqz(preemphasis);

figure("Name","Preenfasis");
subplot(3, 2, 1);
plot(a_preemp);
title('Vocal ''a''');
subplot(3, 2, 2);
plot(o_preemp);
title('Vocal ''o''');
subplot(3, 2, 3);
plot(p_preemp);
title('Letra ''p''');
subplot(3, 2, 4);
plot(s_preemp);
title('Letra ''s''');
subplot(3, 2, 5);
plot(pasa_preemp);
title('Palabra ''pasa''');
subplot(3, 2, 6);
plot(paso_preemp);
title('Palabra ''paso''');
hold on;

% 3.Ventana de hamming

window_size = 512;
hop_size = 170;
hamming_window = hamming(window_size);
a_windowed = buffer(a_preemp, window_size, hop_size) .* hamming_window;
o_windowed = buffer(o_preemp, window_size, hop_size) .* hamming_window;
p_windowed = buffer(p_preemp, window_size, hop_size) .* hamming_window;
s_windowed = buffer(s_preemp, window_size, hop_size) .* hamming_window;
pasa_windowed = buffer(pasa_preemp, window_size, hop_size) .* hamming_window;
paso_windowed = buffer(paso_preemp, window_size, hop_size) .* hamming_window;

figure("Name","Ventaneo");
subplot(3, 2, 1);
plot(a_windowed);
title('Vocal ''a''');
subplot(3, 2, 2);
plot(o_windowed);
title('Vocal ''o''');
subplot(3, 2, 3);
plot(p_windowed);
title('Letra ''p''');
subplot(3, 2, 4);
plot(s_windowed);
title('Letra ''s''');
subplot(3, 2, 5);
plot(pasa_windowed);
title('Palabra ''pasa''');
subplot(3, 2, 6);
plot(paso_windowed);
title('Palabra ''paso''');
hold on;

% 4. Calcular la potencia de la señal para cada bloque
a_power = sum(abs(a_windowed).^2, 1);
o_power = sum(abs(o_windowed).^2, 1);
p_power = sum(abs(p_windowed).^2, 1);
s_power = sum(abs(s_windowed).^2, 1);
pasa_power = sum(abs(pasa_windowed).^2, 1);
paso_power = sum(abs(paso_windowed).^2, 1);



figure("Name","Potencia");
subplot(3, 2, 1);
plot(a_power);
title('Vocal ''a''');
subplot(3, 2, 2);
plot(o_power);
title('Vocal ''o''');
subplot(3, 2, 3);
plot(p_power);
title('Letra ''p''');
subplot(3, 2, 4);
plot(s_power);
title('Letra ''s''');
subplot(3, 2, 5);
plot(pasa_power);
title('Palabra ''pasa''');
subplot(3, 2, 6);
plot(paso_power);
title('Palabra ''paso''');
hold on;
%%
% 8. Encontrar los umbrales para indicar el inicio y el final de las palabras 'pasa' y 'paso'
umbral = 0.002; % Ajustar según la señal
inicio_pasa = find(pasa_power > umbral, 1, 'first');
fin_pasa = find(pasa_power > umbral, 1, 'last');
inicio_paso = find(paso_power > umbral, 1, 'first');
fin_paso = find(paso_power > umbral, 1, 'last');

%%
% Correlacion
orden = 12;
r = zeros(orden,1);
R = zeros(orden,orden);
w = zeros(orden, 71);

%pasa_windowed % 512*71
N = 512;
for block=1:71
    for j=1:orden
        for k=1:orden
            for n=orden+1:N
                if j==1
                    r(k) = r(k) + pasa_windowed(n - k,block) * pasa_windowed(n, block);
                else
                    R(k, j) = R(k, j) + pasa_windowed(n - k,block) * pasa_windowed(n - j, block);
                end

            end
        end    
    end
    r = r / N;
    R = R / N;
    % w(:, block) = R \ r;
    w(:,block) = pinv(R)*r;
end









