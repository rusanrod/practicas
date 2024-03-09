
import numpy as np
import cv2
import math

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


def hough_lines(img_edges, threshold):
    height, width = img_edges.shape
    max_rho = int(math.hypot(height, width))
    theta_values = np.deg2rad(np.arange(-90, 90))

    accumulator = np.zeros((2 * max_rho, len(theta_values)), dtype=np.uint64)

    for y in range(height):
        for x in range(width):
            if img_edges[y, x] == 255:
                for theta_index, theta in enumerate(theta_values):
                    rho = int(x * np.cos(theta) + y * np.sin(theta))
                    accumulator[rho + max_rho, theta_index] += 1

    # Encontramos los índices de los picos en la matriz de acumulación
    rho_idx, theta_idx = np.where(accumulator >= threshold)

    # Dibujamos las líneas encontradas en la imagen original
    lines_img = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
    for i in range(len(rho_idx)):
        rho = rho_idx[i] - max_rho
        theta = theta_values[theta_idx[i]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return lines_img





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
        

def main():
    img = cv2.imread("lines.png")

    # Utilizar tu función canny para obtener los bordes
    img_edges = canny(img)
    shape = img_edges.shape 
    print(shape)
    img_hough = hough_lines(img_edges, 110)
    
    cv2.imshow('original', img)
    cv2.imshow("detected lines", img_hough)
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
