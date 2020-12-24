import numpy as np
import cv2, os
from PIL import Image

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


recognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8,123)


def get_images(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.happy')]

    images = []
    labels = []

    for image_path in image_paths:
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        subject_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(subject_number)
            cv2.imshow("", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels


path = './yalefaces'
images, labels = get_images(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))

image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.happy')]

for image_path in image_paths:
    gray = Image.open(image_path).convert('L')
    image = np.array(gray, 'uint8')
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])

        number_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        if number_actual == number_predicted:
            print("subject{} is Correctly Recognized with confidence {}%".format(number_actual,
                                                                                 round((123-conf)*100/123, 2)))
        else:
            print("subject{} is Incorrect Recognized as {}".format(number_actual, number_predicted))
        cv2.imshow("Recognizing Face", image[y: y + h, x: x + w])
        cv2.waitKey(1000)
