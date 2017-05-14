
import cv2

# 内蔵カメラをで起動（320x240）
cap = cv2.VideoCapture(0)
if cap.isOpened() is False:
    raise("IO Error")
ret = cap.set(3, 640)
ret = cap.set(4, 480)

# 学習済みの顔認識のデータのパス
face_cascade_path = "/Users/nnamm/anaconda3/envs/py35opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
# 学習済みの瞳認識のデータのパス
eye_cascade_path = "/Users/nnamm/anaconda3/envs/py35opencv/share/OpenCV/haarcascades/haarcascade_eye.xml"
# カスケード分類器の特徴量を取得する
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
# 顔に表示される枠の色を指定（白色）
color = (255, 255, 255)

while True:
    # 内蔵カメラから読み込んだキャプチャデータを取得
    ret, frame = cap.read()
    if ret is False:
        continue
    # 画像解析用にモノクロデータ生成
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔認識の実行
    face_rect = face_cascade.detectMultiScale(grayframe, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))

    # 顔が見つかったらcv2.rectangleで顔に白枠を表示する
    if len(face_rect) > 0:
        for frect in face_rect:
            cv2.rectangle(frame, tuple(frect[0:2]), tuple(frect[0:2]+frect[2:4]), color)

            # todo 瞳認識と描画
            # gray = grayframe[tuple(frect[0:2]), tuple(frect[2:4])]
            # colors = frame[tuple(frect[0:2]), tuple(frect[2:4])]
            # eye_rect = eye_cascade.detectMultiScale(gray)
            # for erect in eye_rect:
            #    cv2.rectangle(colors, tuple(erect[0:2]), tuple(erect[0:2]+erect[2:4]), color)

    # 表示
    cv2.imshow('frame', grayframe)

    # ESCキーを押すとループ終了
    key = cv2.waitKey(10)
    if key == 27:
        break

# 内蔵カメラを終了
cap.release()
cv2.destroyAllWindows()

# 参考サイト
# http://opencv.jp/opencv-2.1/cpp/object_detection.html
# http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
# http://rasp.hateblo.jp/entry/2016/01/24/135417
# http://derivecv.tumblr.com/post/68629592456
# https://www.outoutput.com/programming/python-opencv-face-detection-haar-cascademd-classifiers/
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_video_display/py_video_display.html
# http://www.pythonweb.jp/tutorial/string/index11.html
# http://qiita.com/hitomatagi/items/04b1b26c1bc2e8081427

