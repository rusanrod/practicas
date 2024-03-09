import cv2
import numpy as np

def calculate_harris_response(image, k=0.04):
    # Paso 1: Obtener la matriz de segundo momento para cada pixel
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    Ixx = dx**2
    Ixy = dx * dy
    Iyy = dy**2
    
    # Aplicar convolución gaussiana para suavizar los resultados
    Ixx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    Iyy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    
    # Paso 2: Obtener los valores y vectores propios de la matriz
    det_M = Ixx * Iyy - Ixy**2
    trace_M = Ixx + Iyy
    
    # Paso 3: Calcular la respuesta de Harris
    harris_response = det_M - k * (trace_M**2)
    
    return harris_response

def detect_corners(image, threshold=0.001):
    gray = image
    
    # Paso 4: Determinar los píxeles que representan esquinas
    harris_response = calculate_harris_response(gray)
    corner_threshold = threshold * np.max(harris_response)
    corners = np.zeros_like(gray)
    corners[harris_response > corner_threshold] = 255
    return corners

def draw_corners(image, corners):
    corner_image = image
    corner_image[corners > 0] = [0, 0, 255]  # Color rojo
    return corner_image


# Captura de video desde la cámara web
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = detect_corners(gray)
    corner_image = draw_corners(frame, corners)
    cv2.imshow('Corner Detection', corner_image)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
