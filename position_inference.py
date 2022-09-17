from dis import dis
import torch
import cv2
from math import sqrt
from angle_metrics import get_skii_angle


class Yeti:
    def __init__(self):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path="weights/det_2.pt"
        )

    def plot_boxes(self, det, img, line_thickness=3):
        c1, c2 = (int(det[0]) - 10, int(det[1])), (int(det[2]) + 10, int(det[3]))
        color = (0, 0, 255)
        cv2.rectangle(
            img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA
        )
        if((det[2]-det[0]) > (det[3]-det[1])):
            self.draw_text(
                img, "Tilt Detected", (500, 10), (0, 100, 140), (0, 0, 0)
                )


    def get_centroid(self, coords):
        return (coords[0] + (coords[2] / 2), coords[1] + (coords[3] / 2))

    def draw_text(
        self, img, text, pos, text_color=(0, 255, 0), text_color_bg=(0, 0, 0)
    ):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h + 5), text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, y + text_h + 3 - 1),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            text_color,
            2,
        )
        return text_size

    def get_distance_skii(self, dets, frame):
        centroids = []
        for det in dets:
            centroids.append(self.get_centroid(det))
        x1, y1, x2, y2 = (
            centroids[0][0],
            centroids[0][1],
            centroids[1][0],
            centroids[1][1],
        )
        dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist < 200:
            result = "Close"
        else:
            result = "Far"
        self.draw_text(
            frame, "Separation: {}".format(result), (30, 10), (0, 200, 200), (0, 0, 0)
        )
        return frame, dist

    def inference(self, video_path):
        vid = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        while True:
            ret, frame = vid.read()
            if ret == False:
                break

            detections = []
            skii_angles = []
            results = self.model(frame)
            if not results.pandas().xyxy[0].empty:
                for result in results.pandas().xyxy[0].itertuples():
                    detections.append(
                        [int(result[1]), int(result[2]), int(result[3]), int(result[4])]
                    )
                    skii_angles.append(get_skii_angle(frame[int(result[2]):int(result[4]),int(result[1]):int(result[3])]))

            for detection in detections:
                self.plot_boxes(detection, frame, 3)

            if len(detections) == 2:
                frame, distance = self.get_distance_skii(detections, frame)
            
            if len(skii_angles)>1:
                angle = int(sum(skii_angles)/len(skii_angles))
                if angle > 0:
                    self.draw_text(
                        frame, "Angle: {}".format(str(angle)), (30, 50), (0, 200, 200), (0, 0, 0)
                    )
                else:
                    self.draw_text(
                        frame, "Angle: {}".format("NaN"), (30, 50), (0, 200, 200), (0, 0, 0)
                    )

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        vid.release()
        return "[Video Processing Complete]"


if __name__ == "__main__":
    yeti = Yeti()
    yeti.inference("videos/gp1.mp4")
