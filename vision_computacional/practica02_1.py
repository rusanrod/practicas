
import numpy as np
import cv2

Sx = np.array(([1,0,-1],[2,0,-2],[1,0,-1]))
Sy = Sx.transpose()

def get_gaussian_kernel(k,sigma):
    k = k//2
    H = np.zeros((2*k+1, 2*k+1))
    r,c = H.shape
    for i in range(r):
        for j in range(c):
            H[i,j] = 1.0/(2*np.pi*sigma*sigma)*np.exp(-((i-k)**2 + (j-k)**2)/(2*sigma*sigma))
    H /= np.sum(H)
    return H

def get_gradient_mag_ang(Gx, Gy):
    x, y= Gx.shape
    Gm = np.zeros((x,y))
    Ga = np.zeros((x,y))
    for i in range(x):
         for j in range(y):
               Gm[i][j] = np.math.sqrt(Gx[i][j]**2 + Gy[i][j]**2)
               Ga[i][j] = np.math.atan2(Gy[i][j], Gx[i][j])
    return Gm, Ga

def non_maximum_suppression(Gm, Ga):
    M, N = Gm.shape
    suppressed = np.zeros((M, N))

    angle_quantized = (np.round(Ga * 180. / np.pi / 45.) * 45.) % 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            if (0 <= angle_quantized[i,j] < 22.5) or (157.5 <= angle_quantized[i,j] <= 180):
                q = Gm[i, j+1]
                r = Gm[i, j-1]
            elif (22.5 <= angle_quantized[i,j] < 67.5):
                q = Gm[i+1, j-1]
                r = Gm[i-1, j+1]
            elif (67.5 <= angle_quantized[i,j] < 112.5):
                q = Gm[i+1, j]
                r = Gm[i-1, j]
            elif (112.5 <= angle_quantized[i,j] < 157.5):
                q = Gm[i-1, j-1]
                r = Gm[i+1, j+1]

            if (Gm[i,j] >= q) and (Gm[i,j] >= r):
                suppressed[i, j] = Gm[i,j]
            else:
                suppressed[i, j] = 0

    return suppressed

                        
def hysteresis(img, lower_bound, upper_bound):
    M, N = img.shape
    result = np.zeros((M, N), dtype=np.uint8)

    for i in range(M):
        for j in range(N):
            px = img[i, j]
            if px <= lower_bound:
                result[i, j] = 0
            elif px < upper_bound:
                neighborhood = img[max(0, i-1):min(M, i+2), max(0, j-1):min(N, j+2)]
                if (neighborhood >= upper_bound).any():
                    result[i, j] = 255
            else:
                result[i, j] = 255

    return result

def canny(frame):
    gauss = get_gaussian_kernel(2, 1)
    I = cv2.filter2D(src = frame, ddepth = -1, kernel = gauss)
    Gx = cv2.filter2D(src = I, ddepth = -1, kernel = Sx)
    Gy = cv2.filter2D(src = I, ddepth = -1, kernel = Sy)
    Gx_gray = cv2.cvtColor(Gx, cv2.COLOR_BGR2GRAY)
    Gy_gray = cv2.cvtColor(Gy, cv2.COLOR_BGR2GRAY)
    Gm, Ga = get_gradient_mag_ang(Gx_gray, Gy_gray)
    max = non_maximum_suppression(Gm, Ga)
    return hysteresis(max, 40, 160)


def hough_transform(img_edges):
    rows, cols = img_edges.shape
    max_dist = int(np.ceil(np.sqrt(rows**2 + cols**2)))  # Distancia máxima de la imagen al origen
    thetas = np.deg2rad(np.arange(-90, 90))  # Ángulos de -90 a 89 grados
    n_thetas = len(thetas)
    
    # Definir los límites de rho y theta
    rhos = np.linspace(-max_dist, max_dist, max_dist * 2)
    
    # Crear matriz acumuladora H
    accumulator = np.zeros((2 * max_dist, n_thetas), dtype=np.uint64)
    
    # Encontrar píxeles de borde
    edge_pixels = np.argwhere(img_edges != 0)
    
    # Loop a través de los píxeles de borde y votar en la matriz acumuladora
    for i in range(len(edge_pixels)):
        x, y = edge_pixels[i]
        for t_idx in range(n_thetas):
            rho = int(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx])) + max_dist
            accumulator[rho, t_idx] += 1
            
    return accumulator, rhos, np.rad2deg(thetas)

def hough_peaks(accumulator, num_peaks, threshold):
    # Encontrar picos en la matriz acumuladora
    peak_values = []
    peak_coords = []
    accumulator_temp = np.copy(accumulator)
    
    for _ in range(num_peaks):
        max_value = np.amax(accumulator_temp)
        if max_value >= threshold:
            peak_value = max_value
            peak_coord = np.argwhere(accumulator_temp == max_value)[0]
            peak_values.append(peak_value)
            peak_coords.append(peak_coord)
            accumulator_temp[peak_coord[0], peak_coord[1]] = 0
        else:
            break
            
    return peak_values, peak_coords

def draw_lines(img, rhos, thetas, peak_coords):
    for peak in peak_coords:
        rho, theta_idx = peak
        theta = thetas[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rhos[rho]
        y0 = b * rhos[rho]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
# def main():
#     img = cv2.imread("baboon.jpg", cv2.IMREAD_GRAYSCALE)
    
#     # Utilizar tu función canny para obtener los bordes
#     img_edges = canny(img)
    
#     accumulator, rhos, thetas = hough_transform(img_edges)
    
#     # Encontrar picos en la matriz acumuladora
#     threshold = 100
#     num_peaks = 10
#     peak_values, peak_coords = hough_peaks(accumulator, num_peaks, threshold)
    
#     # Dibujar líneas
#     img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     draw_lines(img_color, rhos, thetas, peak_coords)
    
#     cv2.imshow('original', img)
#     cv2.imshow("detected lines", img_color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



def main():
    # img = cv2.imread("lines.png")

    # Utilizar tu función canny para obtener los bordes
    # img_edges = canny(img)
    
    img = cv2.imread('lines.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(gray,0,255,apertureSize = 3)

    lines = cv2.HoughLines(img_edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    # cv2.imwrite('houghlines1.jpg',img)
    
    cv2.imshow('original', img_edges)
    cv2.imshow("detected lines", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Canny algorith using video capture is too slow (dont recommended)
    '''cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow("original", frame)
        new = canny(frame)
        cv2.imshow("blurred", new )

        if cv2.waitKey(10) & 0xFF == 27:
            # cv2.waitKey(0)
            break
        # print(frame.shape)
    cap.release()
    cv2.destroyAllWindows()'''

if __name__ == '__main__':
    main()
