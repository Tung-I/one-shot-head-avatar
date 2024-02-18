import cv2
import os
from mtcnn import MTCNN


SRC_DIR = '/home/ubuntu/datasets/cap/src'
SRC_FNAME = 'source.png'

class LandmarkDetector():
    """
    Preprocess the source image for the GOHA model
    - MTCNN face landmark detection
    """
    
    def __init__(self):
        self.src_dir = SRC_DIR

        # MTCNN
        self.detector = MTCNN()

    def mtcnn(self, fname):
        """
        MTCNN face detection:
         - detect faces and save the landmarks to src_dir/detections
        """
        image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        result = self.detector.detect_faces(image)

        if len(result)>0:
            index = 0
            if len(result)>1: # if multiple faces, take the biggest face
                size = -100000
                for r in range(len(result)):
                    size_ = result[r]["box"][2] + result[r]["box"][3]
                    if size < size_:
                        size = size_
                        index = r

            bounding_box = result[index]['box']
            keypoints = result[index]['keypoints']
            if result[index]["confidence"] > 0.9:

                dst = fname.replace('images', 'detections').replace('.png', '.txt')
                print(f'Save to MTCNN output to: {dst}')
                outLand = open(dst, "w")
                outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['nose'][0])) + " " +      str(float(keypoints['nose'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
                outLand.close()

def main():
    model = LandmarkDetector()
    src_fname = os.path.join(SRC_DIR, SRC_FNAME)

    # MTCNN
    model.mtcnn(src_fname) 

if __name__ == "__main__":
    main()