import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json, load_model

# import facial

cvSocket = None

dir = os.path.dirname(__file__) + '/models/'


class Predictor(object):
	def __init__(self):
		self.emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
		self.people = {0:"Raghav", 1:"Varun", 2:"Shivam", 3:"Akhila", 4:"Nihal"}
		self.holdingArray = []
		self.namesArray = []
		self.currentEmotion = None
		self.currentName = None

		self.face_cascade = cv2.CascadeClassifier(os.path.join(dir, 'face_features.xml'))
		self.emotion_model = model_from_json(open(os.path.join(dir, 'facial_expression_model_structure.json'), 'r').read())
		self.emotion_model.load_weights(os.path.join(dir, 'facial_expression_model_weights.h5'))
		self.emotion_model._make_predict_function()

		# name_model = load_model('models/name_model3.h5')
		# face_recognizer = facial.setup()

	@staticmethod
	def _most_frequent_element(array):
		n = len(array)
		if n == 0: return None, None

		array = sorted(array)
		max_count = 1; res = array[0]; curr_count = 1

		for i in range(1, n):
			if array[i] == array[i - 1]:
				curr_count += 1
			else:
				if curr_count > max_count:
					max_count = curr_count
					res = array[i - 1]
				curr_count = 1

		if curr_count > max_count:
			max_count = curr_count
			res = array[n - 1]
		return res, max_count

	def predict_emotion(self, input_img):
		img = cv2.resize(np.array(input_img), (480, 360))
		img = img[0:308,:]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

		for (x,y,w,h) in faces:
			if w > 100: #trick: ignore small faces
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5) #highlight detected face
				detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
				detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
				detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

				# name_pred = facial.predict(detected_face, face_recognizer)
				# name = (self.people[name_pred[0]])
				# if len(nameArray) < 10:
				# 	nameArray.append(name)
				# else:
				# 	del nameArray[0]
				# 	nameArray.append(name)
				# name = self.namesArray[1]

				face_pixels = image.img_to_array(detected_face)
				face_pixels = np.expand_dims(face_pixels, axis = 0)
				face_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

				emotion_preds = self.emotion_model.predict(face_pixels) #store probabilities of 7 expressions

				#background of expression list
				overlay = img.copy()
				opacity = 0.4
				# cv2.rectangle(img,(x+w+10,y-25),(x+w+150,y+115),(0,64,0),cv2.FILLED)
				# cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

				#connect face and expressions
				#TODO Name on top of emotions
				# cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
				# cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
				# cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255,255,255),1)

				emotionArr = []
				for i in range(len(emotion_preds[0])):
					emotionArr.append((self.emotions[i], round(emotion_preds[0][i]*100, 2)))
					emotion = "%s %s%s" % (self.emotions[i], round(emotion_preds[0][i]*100, 2), '%')
					color = (255,255,255)
					# cv2.putText(img, emotion, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

				emotion = max(emotionArr, key=lambda item:item[1])[0]

				if len(self.holdingArray) < 10:
					self.holdingArray.append(emotion)
				else:
					del self.holdingArray[0]
					self.holdingArray.append(emotion)

				max_emotion, _ = Predictor._most_frequent_element(self.holdingArray)
				user = Predictor._most_frequent_element(self.namesArray)

				img_cv = cv2.resize(img, (400, 300))

				return img_cv, max_emotion, "User"

		return img, None, "User"
