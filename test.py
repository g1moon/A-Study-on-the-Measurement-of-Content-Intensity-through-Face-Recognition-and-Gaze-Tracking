from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import pandas as pd
from gaze_tracking import GazeTracking
from time import localtime, strftime
import dbConn as db
import math


t = time.time()

df = {'contentsId'     : []
    ,'frameId'  : []
    ,'age' : []
    ,'gender'   : []
    ,'freqId'      : []
    ,'is_gazing'  : []

}
person_info = pd.DataFrame(df)





def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (0, 0, 255), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 50,200 ), 2)
    
         #90, 60
##
@contextmanager
def video_capture(*args, **kwargs):
    cap = VideoStream(src=0).start()
    try:
        yield cap
    finally:
        cap.stop()

def yield_images():
    # capture video
    with video_capture(0) as cap:
#        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:

            # get video frame
            img = cap.read()
            img = imutils.resize(img, width=600)
#
#            if not ret:
#                raise RuntimeError("Failed to capture image")

            yield img




def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    #
    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            #?
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

def main():
    time.sleep(1.0)
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir
    gaze = GazeTracking()
    started_time = strftime('%Y-%m-%d %I:%M:%S')
    




    ct = CentroidTracker()
    (img_h, img_w) = (None, None)
    
    net = cv2.dnn.readNetFromCaffe('./model/deploy.prototxt', './model/res10_300x300_ssd_iter_140000.caffemodel')

    
    if not weight_file:
    

        weight_file = './model/dafarm_weight.hdf5'

    detector = dlib.get_frontal_face_detector()


    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    
    

    for img in image_generator:
        
        t = time.time()
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)
        
        blob = cv2.dnn.blobFromImage(img, 1.0, (img_w, img_h),
                                     (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []

        detected = detector(input_img, 1)
    
    

        faces = np.empty((len(detected), img_size, img_size, 3))
        
        


        if len(detected) > 0:
            for i, d in enumerate(detected):
                
                
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                rects.append([x1,y1,x2,y2])
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
                
                objects = ct.update(rects)
                

            for (objectID, centroid) in objects.items():
                    

                gaze.refresh(img)
        #
                img = gaze.annotated_frame() #
                global text



                if gaze.is_blinking():
                    text = '0'
                elif gaze.is_right():
                    text = '0'
                elif gaze.is_left():
                    text = '0'
                elif gaze.is_center():
                    text = 'gazing'


            print(person_info.tail(5))
#            print(person_ifno.tail(1))


            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.50 else "F")
                draw_label(img, (d.left(), d.top()), label)
                global freqId
                person_info.loc[len(person_info) + 1] = ['contents_1'         # 역 이름
                                                         ,i
                                                         ,predicted_ages[i]
                                                         ,0 if predicted_genders[i][0] < 0.50 else 1
                                                         ,'freqId'
                                                         ,text
                                                         ]

        cv2.imshow("result", img)
        cv2.imshow('result', img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        

        if key == 27:

            
            finished_time = strftime('%Y-%m-%d %I:%M:%S')

#            person_info['gaze_time'].astype('float')
#
#            age= round(person_info.groupby(person_info['frameId'])['age'].mean())
#            gaze_time=person_info.groupby(person_info['frameId'])['gaze_time'].sum()
#            freqId =person_info.groupby(person_info['frameId'])['age'].count()
#            gender=person_info.groupby(person_info['frameId'])['gender'].mean()
##
#            for gen in person_info['frameId'].unique():
#                if gender[gen] < 0.5:
#                    gender[gen] = 'M'
#                else:
#                    gender[gen] = 'F'


            person_info1 = pd.DataFrame(df)
            person_info1['is_gaze'] = text

#            db.Conn('./dafarm/testDB.sqlite')
#            for i in range(0, len(freqId)) :
#                args = [gender.values[i], age.values[i], gaze_time.values[i], '역이름', '6']
#                #print("log :", age.values[0])
#
#                db.InsertLog(args)
#            db.DisConn()
#
#            person_info1.drop(['frameId'], axis = 1, inplace =True)
#            person_info1= person_info1.reset_index()
#
            person_info.to_csv('./data/' + started_time +'~' + finished_time+ '_person_info', mode = 'w')
#             ESC
            break


if __name__ == '__main__':
    main()

    



