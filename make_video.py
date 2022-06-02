import cv2
import sys

object_name_list = ['glasses_1', 'glasses_3', 'earring_1', 'earring_2']

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
img = cv2.imread(f'images/img_out/glasses_1/face_0.jpg')
out = cv2.VideoWriter('result_video.mp4', fourcc, 20.0, (img.shape[1], img.shape[0]))

for object_name in object_name_list:
	for img_id in range(135):
		img = cv2.imread(f'images/img_out/{object_name}/face_{img_id}.jpg')
		out.write(img)
out.release()