from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

import time
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QLabel, QPushButton, QProgressBar, \
    QRadioButton, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

pretrained_model = "pretrained_models/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

class MyWidget(QWidget):
    def __init__  (self):
        QWidget. __init__ (self)
    myclose = True

    def closeEvent(self,event):
        if self.myclose:
            print(self.myclose)
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                print("")
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWidget()
    w.resize(330,210)
    w.setWindowTitle('AgeGender')

    pbar = QProgressBar(w)
    pbar.setGeometry(10, 165, 290, 30)


    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.8, thickness=1):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    @contextmanager
    def video_capture(*args, **kwargs):
        cap = cv2.VideoCapture(*args, **kwargs)
        try:
            yield cap
        finally:
            cap.release()

    def yield_images():
        # capture video
        with video_capture(0) as cap:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while True:
                # get video frame
                ret, img = cap.read()

                if not ret:
                    raise RuntimeError("Failed to capture image")

                yield img

    def yield_images_from_dir(image_dir):
        image_dir = Path(image_dir)

        for image_path in image_dir.glob("*.*"):
            img = cv2.imread(str(image_path), 1)

            if img is not None:
                h, w, _ = img.shape
                r = 640 / max(w, h)
                yield cv2.resize(img, (int(w * r), int(h * r)))

    def count_im_dir(image_dir):
        image_dir = Path(image_dir)
        im_count = 0
        for image_path in image_dir.glob("*.*"):
            im_count += 1
        return im_count

    def progBarUpdate(percent):
        pbar.setValue(percent)

    def startRec():
        depth = 16
        k = 8
        margin = 0.4
        image_dir = nameEdit.text()

        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                                   file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

        # for face detection
        detector = dlib.get_frontal_face_detector()

        # load model and weights
        img_size = 64
        model = WideResNet(img_size, depth=depth, k=k)()
        model.load_weights(weight_file)

        image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
        count_im_in_dir = count_im_dir(image_dir)

        count = 0
        for img in image_generator:
            #print(count)
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

                # predict ages and genders of the detected faces
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                # draw results
                for i, d in enumerate(detected):
                    label = "{}, {}".format(int(predicted_ages[i]),
                                            "M" if predicted_genders[i][0] < 0.5 else "F")
                    draw_label(img, (d.left(), d.top()), label)

            #cv2.imshow("result", img)
            cv2.imwrite(name1Edit.text() + "/res" + str(count) + ".jpg", img)
            count += 1
            progBarUpdate(100 * (count / count_im_in_dir))
            key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()





    dirLabel = QLabel(w)
    dirLabel.setText("Директория с исходными изображениями:")
    dirLabel.move(10,10)
    dirLabel.show()

    nameEdit = QLineEdit(w)
    nameEdit.move(10,40)
    nameEdit.show()

    dir1Label = QLabel(w)
    dir1Label.setText("Целевая директория:")
    dir1Label.move(10,70)
    dir1Label.show()

    name1Edit = QLineEdit(w)
    name1Edit.move(10,100)
    name1Edit.show()

    button = QPushButton(w)
    button.setText('Обработать')
    button.move(10,130)
    button.show()
    button.clicked.connect(startRec)

    w.show()
    sys.exit(app.exec_())
