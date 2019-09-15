#import OpenCV module
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
#function to detect face
def detect_face (img):
    #convert the test image to gray image
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector
    face_cas = cv2.CascadeClassifier ('models/face_features.xml')
    faces = face_cas.detectMultiScale (gray, scaleFactor=1.3, minNeighbors=4);
    #if no faces are detected then return image
    if (len (faces) == 0):
        return None, None
    #extract the face
    faces [0]=(x, y, w, h)
    #return only the face part
    return gray[y: y+w, x: x+h], faces [0]
#this function will read all persons' training images, detect face #from each image
#and will return two lists of exactly same size, one list
def prepare_training_data(data_folder_path):
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
    #our subject directories start with letter 's' so
    #ignore any non-relevant directories if any
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        if (dir_name == '.DS_Store'):
            continue
        label = int(dir_name)
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        #------STEP-3--------
        #go through each image name, read image,
        #detect face and add face to list of faces
        for image_name in subject_images_names:
        #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
        #build image path
        #sample image path = training-data/s1/1.pgm
        image_path = subject_dir_path + "/" + image_name
        #read image
        image = cv2.imread(image_path)
        #detect face
        # face, rect = detect_face(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48)) #resize to 48x48

        face = image
        #------STEP-4--------
        #we will ignore faces that are not detected
        if face is not None:
            #add face to list of faces
            faces.append(face)
            #add label for this face
            labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels


def predict(test_img, face_recognizer):
#make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    face = img
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    label= face_recognizer.predict(face)
    return label


def setup():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Preparing data...")
    faces, labels = prepare_training_data("./training-data")
    print("Data prepared")
    #print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    #create our LBPH face recognizer
    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))
    # test_img1 = cv2.imread("test-data/Shivam_6.jpg")
    # test_img2 = cv2.imread("test-data/Varun_6.jpg")
    # predicted_img1 = predict(test_img1)
    # predicted_img2 = predict(test_img2)
    # pred = predict(image)
    # print("Prediction complete")
    return face_recognizer
