import torch
import cv2
from math import sqrt

class Yeti:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path="weights/det_2.pt")

    def plot_boxes(self, det, img, line_thickness=3):
        c1, c2 = (int(det[0]) - 10, int(det[1])), (int(det[2]) + 10, int(det[3]))
        color = (0, 0, 255)
        cv2.rectangle(
            img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA
        )

    def get_centroid(self,coords):
        return (coords[0]+(coords[2]/2), coords[1]+(coords[3]/2))
    
    def get_distance_skii(self,dets):
        centroids = []
        for det in dets:
            centroids.append(self.get_centroid(det))
        x1,y1,x2,y2 = centroids[0][0],centroids[0][1],centroids[1][0],centroids[1][1]
        dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        return dist

    def inference(self,video_path):
        vid = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
        
        while True:
            ret, frame = vid.read()
            if ret == False:
                break
            
            detections = []
            results = self.model(frame)
            if not results.pandas().xyxy[0].empty:
                        for result in results.pandas().xyxy[0].itertuples():
                            print(result)
                            detections.append(
                                    [
                                        int(result[1]),
                                        int(result[2]),
                                        int(result[3]),
                                        int(result[4])
                                    ]
                                )
            
            for detection in detections:
                self.plot_boxes(detection,frame,3)
            
            if len(detections) == 2:
                distance = self.get_distance_skii(detections)

            cv2.imshow("frame",frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        vid.release()
        return("[Video Processing Complete]")
    
if __name__ == "__main__":
    yeti = Yeti()
    yeti.inference('videos/gp1.mp4')
