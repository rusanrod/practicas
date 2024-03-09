import numpy as np

# Paso 1: Leer los archivos CSV y cargarlos en matrices de N x 3
points1 = np.loadtxt('Points1.csv', delimiter=',')
points2 = np.loadtxt('Points2.csv', delimiter=',')

# Paso 2: Obtener la matriz de covarianza para cada conjunto de datos
covariance_matrix1 = np.cov(points1, rowvar=False)
covariance_matrix2 = np.cov(points2, rowvar=False)


# Paso 3: Obtener los valores y vectores propios de las matrices de covarianza
eigenvalues1, eigenvectors1 = np.linalg.eig(covariance_matrix1)
eigenvalues2, eigenvectors2 = np.linalg.eig(covariance_matrix2)

# Paso 4: Determinar si se trata de una recta, un plano, o ninguno de los dos
def determine_geometry(eigenvalues, umbral = 0.5):
    # Normalizar los valores
    valores_abs = np.abs(eigenvalues)

    # Calcular el logaritmo de los valores normalizados
    log_valores = np.log10(valores_abs + 1e-10)  # Evitar log(0)

    # Calcular la diferencia entre los logaritmos
    log_diff = np.abs(log_valores - np.mean(log_valores))

    # Determinar si están en la misma magnitud
    if np.all(log_diff <= umbral):
        return "Plano"
    elif np.sum(valores_abs / np.max(valores_abs) < umbral) == 1:
        return "Plano"
    elif np.sum(valores_abs / np.max(valores_abs) < umbral) == 2:
        return "Recta"
    else:
        return "Ninguno"

geometry1 = determine_geometry(eigenvalues1)
geometry2 = determine_geometry(eigenvalues2)

print("Geometry of Points1.csv:", eigenvalues1, geometry1)
print("Geometry of Points2.csv:", eigenvalues2,geometry2)

# Paso 5: Obtener la ecuación que describe la recta o el plano
# Las rectas se describen en su forma parametrica y los planos con los coeficientes de la ecuacion implicita

def get_equation(eigenvalues, eigenvectors, geometry, point ):
    if geometry == "Plano":
        # min_eig_val_index = eigenvalues.index(min(eigenvalues))
        min_eig_val_index = np.where(eigenvalues == min(eigenvalues))[0][0]
        norm_vect = eigenvectors[:,min_eig_val_index]
        a,b,c = norm_vect
        x,y,z = point
        d = -a*x -b*y -c*z
        return [a,b,c,d]
    
    elif geometry == 'Recta':
        max_eig_val_index = np.where(eigenvalues == max(eigenvalues))[0][0]
        dir_vect = eigenvectors[:, max_eig_val_index]
        return [point, dir_vect]

mean_pts_1 = np.mean(points1, axis=0)
mean_pts_2 = np.mean(points2, axis=0)

equation1 = get_equation(eigenvalues1, eigenvectors1, geometry1, mean_pts_1)
equation2 = get_equation(eigenvalues2, eigenvectors2, geometry2, mean_pts_2)

print("Equation for Points1.csv:", equation1)
print("Equation for Points2.csv:", equation2)

# Paso 6: Leer el archivo Prism.csv y cargar los datos en una matriz de N x 3
prism_data = np.loadtxt('Prism.csv', delimiter = ',')

# Paso 7: Obtener los valores y vectores propios de la matriz de covarianza
covariance_matrix_prism = np.cov(prism_data, rowvar=False)
eigenvalues_prism, eigenvectors_prism = np.linalg.eig(covariance_matrix_prism)

# Paso 8: Construir la matriz T utilizando los vectores propios de la matriz de covarianza
T = eigenvectors_prism.T

# Paso 9: Transformar todos los puntos al nuevo sistema de coordenadas
transformed_prism_data = np.dot(prism_data, T)

# Paso 10: Determinar las dimensiones del prisma definido por los puntos contenidos en el archivo Prism.txt
dim = [max(transformed_prism_data[:,0]) - min(transformed_prism_data[:,0]),
        max(transformed_prism_data[:,1])- min(transformed_prism_data[:,1]), 
        max(transformed_prism_data[:,2])- min(transformed_prism_data[:,2])]
print("Dimensions of the prism:", dim)
