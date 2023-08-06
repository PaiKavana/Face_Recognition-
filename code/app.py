from flask import Flask, render_template, request
from PIL import Image
import io
import face_recognition as fr
import os,cv2
import numpy as np

app = Flask(__name__)

def encode_faces(folder) :
    encoding_list = []

    for filename in os.listdir(folder) :
        known_img = fr.load_image_file(f'{folder}/{filename}')
        known_enc = fr.face_encodings(known_img)[0]
        encoding_list.append((known_enc,filename))

    return encoding_list

def create_frame(location,label,img,names) :
    top,right,bottom,left = location

    label = label.replace('.png','')
    label = label.replace('.jpg','')
    names.append(label)
    cv2.rectangle(img,(left,top),(right,bottom),(26, 237, 30),2)
    cv2.rectangle(img,(left,bottom+20),(right,bottom),(26, 237, 30),-1)
    cv2.putText(img,label,(left+3,bottom+14),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    return names


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
    UPLOAD_FOLDER = 'Images'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if 'image' not in request.files:
        return 'No file part in the request'

    file = request.files['image']
    if file.filename == '':
        return 'No file selected'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    image = cv2.imread(file_path)
    resized_image = cv2.resize(image, (500, 500))
    cv2.imwrite(file_path, resized_image)

    return 'File uploaded and processed successfully'

@app.route('/compare', methods=['POST'])
def compare():
    target_image = request.files['image']
    pred=['Face Detected','Face not Detected']
    names=[]
    target_image=fr.load_image_file(target_image)
    target_encoding=fr.face_encodings(target_image)

    face_loc=fr.face_locations(target_image)
    for person in encode_faces('Images'):
        encode_face=person[0]
        filename=person[1]

        is_target_face=fr.compare_faces(encode_face,target_encoding,tolerance=0.6)
        print(f'{is_target_face} {filename}')

        if face_loc:
            face_no=0
            for location in face_loc:
                if is_target_face[face_no]:
                    label=filename

                    #create_frame(location,label)
                    label = label.replace('.png','')
                    label = label.replace('.jpg','')
                    names.append(label)
                face_no=face_no+1

    
    if names:
        text="face Detected"
        for i in names:
            text=text+"\n"+i
    else:
        text="face not detected"


    return render_template('result.html',text=text)

if __name__ == '__main__':
    app.run(debug=True)
