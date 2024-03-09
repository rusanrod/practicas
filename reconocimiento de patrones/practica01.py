import random
import numpy as np
import matplotlib.pyplot as plt

# 1. Genere la señal x(n) = 0.6530 x(n-1) - 0.7001 x(n-2) + v(n)
# x(-1) = 0, x(-2) = 0, v(n): ruido blanco con media 0

# graficar la señal con 2000 puntos
N = 2000
x = np.zeros((N,1))

for n in range(2, len(x)):
    vn = random.uniform(-1,1) # .gauss(mu=0.0, sigma=1.0)
    x[n] = 0.6230 * x[n-1] - 0.7007 * x[n-2] + vn

# En realidad se grafica al final para que el programa termine su ejecucion
#plt.plot(x)
#plt.show()

# 2. Encuentre los coeficientes w1 y w2 de un filtro predictor de x(n) utilizando el filtro de Wiener

r = np.zeros((2,1))
R = np.zeros((2,2))

for j in range(0,3):
    for k in range(1,3):
        for n in range(2, N):            
            if j == 0:
                r[k-1] += x[n-k] * x[n]
            
            else:
                R[k-1, j-1] += x[n-k] * x[n-j]

r = r * 1/N
R = R * 1/N
w = np.matmul(np.linalg.inv(R), r)
# print("r:", r)
# print("R:", R)
#print("Los coeficientes calculados con filtro de wiener son:", w)

# 3 Encuentre los coeficientes w1 y w2 de un filtro predictor x(n)
# utilizando la tecnica de filtros adaptables LMS

mu = 0.01
w_LMS = np.zeros((2,1))
e = np.zeros((N,1))
w_values = []

for n in range(2, N):
    un1 = [x[n-1], x[n-2]]
    y = w_LMS[0] * un1[0] + w_LMS[1] * un1[1]
    e[n] = x[n] - y
    w_LMS = w_LMS + mu * e[n] * un1
    w_values.append(w_LMS.flatten())

w_LMS_values = np.array(w_values)

filas = 5
# Graficar x, abs(e), w1 y w2 en ese orden
plt.figure(figsize=(10, 6))

plt.subplot(filas, 1, 1)
plt.plot(x, label='x')
plt.xlabel('Muestra')
plt.ylabel('Valor')
plt.title('Señal x')
plt.xlim(0, N)

plt.subplot(filas, 1, 2)
plt.plot(np.abs(e), label='|e|', color='orange')
plt.xlabel('Muestra')
plt.ylabel('Valor')
plt.title('Magnitud del error absoluto (|e|)')
plt.xlim(0, N)

plt.subplot(filas, 1, 3)
plt.plot(w_LMS_values[:, 0], label='w_LMS[0]', color='green')
plt.xlabel('Iteración')
plt.ylabel('Valor')
plt.title('Valor de w_LMS[0] en cada iteración')
plt.xlim(0, N)

plt.subplot(filas, 1, 4)
plt.plot(w_LMS_values[:, 1], label='w_LMS[1]', color='red')
plt.xlabel('Iteración')
plt.ylabel('Valor')
plt.title('Valor de w_LMS[1] en cada iteración')
plt.xlim(0, N)

plt.subplot(filas, 1 ,5)
plt.text(0.1, 0.9, 'w_real: {}, {}'.format(0.6530, -0.7001), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.1, 0.6, 'w_wiener: {}, {}'.format(w[0,0], w[1,0]), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.1, 0.3, 'w_LMS: {}, {}'.format(w_LMS_values[-1][0], w_LMS_values[-1][1]), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()