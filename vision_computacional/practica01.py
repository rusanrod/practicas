import numpy as np
import cv2

# captura de video de la webcam 
cap = cv2.VideoCapture(0)

# Dibujar el bounding box desde la posicion inicial del mouse hasta la posicion actual o final
def draw_bb(frame, start, end):
	frame_copy = frame.copy()
	cv2.rectangle(frame_copy, start, end, (255,0,0), 2)
	cv2.imshow("Video raw", frame_copy)

# Realiza el promedio del bounding box seleccionado, obtiene los limites superior e inferior de valores a ser filtrados
# En caso de no seleccionar un area retira los limites para eliminar el filtro
def get_mask_params(start, end):
	y_min, y_max = min(start[1], end[1]), max(start[1], end[1])
	x_min, x_max = min(start[0], end[0]), max(start[0], end[0])

	cropped_frame = frame[y_min:y_max, x_min:x_max]
	if len(cropped_frame) > 0:
		bgr_mean = tuple(map(int, cv2.mean(cropped_frame)))[:3]
		th_array = np.array((threshold, threshold, threshold))
		lower_bound = np.array(bgr_mean) - th_array
		upper_bound = np.array(bgr_mean) + th_array
		return (lower_bound, upper_bound)
	else:
		return [0,0]

# Genera la mascara tanto en el frame de video y el opuesto en la imagen background
def apply_mask(frame, bounds, background):
	mask = cv2.inRange(frame, bounds[0], bounds[1])
	mask_back = cv2.bitwise_not(mask)
	if mask is not None:
		coincidence = cv2.bitwise_and(frame, frame, mask=mask_back)
		background_not = cv2.bitwise_and(background, background, mask = mask)

		backgrounded = cv2.bitwise_or(coincidence, background_not)
		cv2.imshow("Video filtered", backgrounded)
		# cv2.imshow("Video backgrounded", backgrounded)


# Eventos del mouse:
		#  click down: guarda la posicion del click y declara el inicio de creacion de bounding box
		#  click up: guarda la posicion final del bounding box y declara el final de la creacion del mismo
		#  mouse move: guarda la posicion actual de mouse para dibujarlo en el frame actual
def mouse_callback(event, x, y, flags, param):
	global create_bb, bb_start_pos, bb_end_pos, bounds
	if event == cv2.EVENT_LBUTTONDOWN:
		bb_start_pos = (x,y)
		create_bb = True

	elif event == cv2.EVENT_LBUTTONUP:
		bb_end_pos = (x,y)
		create_bb = False
		bounds = get_mask_params(bb_start_pos, bb_end_pos)

	elif event == cv2.EVENT_MOUSEMOVE:
		bb_end_pos = (x,y)
	
	if create_bb:
		draw_bb(frame, bb_start_pos, bb_end_pos)

def trackbar_callback(value):
	global threshold
	threshold = value


def main():
	global threshold, create_bb, frame, bb_start_pos, bb_end_pos, bounds
	create_bb = False
	threshold = 10
	bounds = [0,0]
	cv2.namedWindow('Video raw')
	cv2.setMouseCallback("Video raw", mouse_callback)
	cv2.createTrackbar('r', "Video raw", threshold, 100, trackbar_callback)

	cv2.namedWindow("Video filtered")
	
	background = cv2.imread("background.jpg")
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break

		if not create_bb:
			cv2.imshow("Video raw", frame)
		else:
			draw_bb(frame, bb_start_pos, bb_end_pos)

		apply_mask(frame, bounds, background)
		
		if cv2.waitKey(10) & 0xFF == 27:
			break
		#print(frame.shape)
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()