from PIL import Image
import face_recognition as FR 
import tensorflow as tf 
import matplotlib.pyplot as plt

def imageProcess():

	image = FR.load_image_file("images/Mac.png")
	print(type(image))
	print(image.shape)
	# print(image)

def faceNum():
	image = FR.load_image_file("images/Mac.png")
	face_locations = FR.face_locations(image)
	print("Face Number: {}".format(len(face_locations)))
	print(type(face_locations))
	print(face_locations)




def result():
	image = FR.load_image_file("images/Mac.png")
	face_locations = FR.face_locations(image)
	print("Face Number: {}".format(len(face_locations)))
	for face_location in face_locations:
		top, right, bottom, left = face_location
		print("Face Location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top,left,bottom,right))
		face_image = image[top:bottom, left:right]
		pil_image = Image.fromarray(face_image)
		pil_image.show()

with tf.Session() as sess:
	image = FR.load_image_file("images/Mac.png")
	face_locations = FR.face_locations(image)
	print(image.shape)
	image_height, image_width, dimensions = image.shape
	print(image_height)
	print("Face Number: {}".format(len(face_locations)))
	for face_location in face_locations:
		top, right, bottom, left = face_location
		# [(39, 225, 168, 96)]
		print(face_locations)
		xmin, ymax, xmax, ymin = face_location
		loc = [xmin/image_height, ymin/image_width, xmax/image_height, ymax/image_width]
		print(loc)
		
		print("Face Location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top,left,bottom,right))
		face_image = image[top:bottom, left:right]
		pil_image = Image.fromarray(face_image)
		# pil_image.show()

	# (309, 389, 3)
	# Face Location: Top: 39, Left: 96, Bottom: 168, Right: 225
	# box_process = 
		batched = tf.expand_dims(tf.image.convert_image_dtype(image, tf.float32),0)
		boxes = tf.constant([[loc]])
		bounding_box = tf.image.draw_bounding_boxes(batched, boxes,name="Hello")
		plt.imshow(bounding_box[0].eval())
		plt.show()







# imageProcess()
# faceNum()
# result()