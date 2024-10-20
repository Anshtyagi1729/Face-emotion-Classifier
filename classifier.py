import cv2
import torch
from torchvision.models import resnet18,ResNet18_Weights
import torchvision.transforms as transforms
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap

model=resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc=torch.nn.Linear(512,7)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

emotion_labels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

app=QApplication([])
window=QWidget()
layout=QVBoxLayout()

video_feed_label=QLabel()
layout.addWidget(video_feed_label)

window.setLayout(layout)
window.show()

face_cascade=cv2.CascadeClassifier('D:\deskapp\haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    for(x,y,w,h) in faces:
        face_roi_color=frame[y:y+h,x:x+h]
        preprocessed_face=preprocess(face_roi_color).unsqueeze(0)
        with torch.no_grad():
            emotion_prediction=model(preprocessed_face)
            predicted_emotion_idx=torch.argmax(emotion_prediction,dim=1).item()
            emotion_label=emotion_labels[predicted_emotion_idx]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,emotion_label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            video_feed_label.setPixmap(QPixmap.fromImage(qt_image))
            app.processEvents()
            
            if cv2.waitKey(1)& 0xFF == ord('q'):
                break
            
cap.release()
cv2.destroyAllWindows()
app.quit()