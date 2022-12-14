from re import S
import torch
import numpy as np
import cv2
import pandas as pd
from math import sqrt
from angle_metrics import get_skii_angle
from mono_depth import MonoDepth


class Yeti:
    def __init__(self):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path="weights/det_2.pt"
        )
        self.mono = MonoDepth()
        self.sensor_data = tuple()
        self.collected_data = []
        self.angle = 0
        self.split = ""
        self.depth = ""

    def plot_boxes(self, det, img, line_thickness=3):
        c1, c2 = (int(det[0]) - 10, int(det[1])), (int(det[2]) + 10, int(det[3]))
        color = (0, 0, 255)
        cv2.rectangle(
            img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA
        )
        if (det[2] - det[0]) > (det[3] - det[1]):
            self.draw_text(img, "Tilt Detected", (500, 10), (0, 100, 140), (0, 0, 0))

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

    def sensor_data_fetch(self):
        df = pd.read_csv(r"pod01_raw.csv")
        data_df = []
        for index, rows in df.iterrows():
            data_df.append([rows.Latitude, rows.Longitude])
        self.sensor_data = data_df

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
            self.split = "Close"
        else:
            result = "Far"
            self.split = "Far"
        self.draw_text(
            frame, "Separation: {}".format(result), (30, 10), (0, 200, 200), (0, 0, 0)
        )
        return frame, dist

    def map_depth_values(self, frame, depth):
        self.depth = depth
        self.draw_text(
            frame,
            "Depth: {}".format(str(depth)),
            (30, 80),
            (0, 200, 200),
            (0, 0, 0),
        )

    def prep_data(self, data):
        self.collected_data.append(
            {
                "gpscoords": str(data[0][0]) + "," + str(data[0][1]),
                "angle": str(data[1]),
                "split": self.split,
                "depth": self.depth,
            }
        )
        return "OK"

    def inference(self, video_path):
        vid = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        frame_count = 0
        dataframe_count = 0
        while True:
            ret, frame = vid.read()
            if ret == False:
                break

            frame_count += 1
            if frame_count % 3 == 0:
                dataframe_count += 1

            relative_dataset = self.sensor_data[dataframe_count]
            detections = []
            skii_angles = []
            results = self.model(frame)
            if not results.pandas().xyxy[0].empty:
                for result in results.pandas().xyxy[0].itertuples():
                    detections.append(
                        [int(result[1]), int(result[2]), int(result[3]), int(result[4])]
                    )
                    skii_angles.append(
                        get_skii_angle(
                            frame[
                                int(result[2]) : int(result[4]),
                                int(result[1]) : int(result[3]),
                            ]
                        )
                    )

            if len(detections) == 2:
                frame, distance = self.get_distance_skii(detections, frame)

            if len(detections) > 0:
                depth_image, approx_depth = self.mono.get_depth(frame, detections[0])
                frame = np.concatenate((frame, depth_image), axis=1)
                self.map_depth_values(frame, int(approx_depth[2]))
            if len(detections) == 0:
                depth_image, approx_depth = self.mono.get_depth(
                    frame, [100, 300, 400, 300]
                )
                frame = np.concatenate((frame, depth_image), axis=1)
                self.map_depth_values(frame, "NaN")

            for detection in detections:
                self.plot_boxes(detection, frame, 3)

            if len(skii_angles) > 1:
                self.angle = int(sum(skii_angles) / len(skii_angles))
                if self.angle > 0:
                    self.draw_text(
                        frame,
                        "Angle: {}".format(str(self.angle)),
                        (30, 50),
                        (0, 200, 200),
                        (0, 0, 0),
                    )
                else:
                    self.angle = 0
                    self.draw_text(
                        frame,
                        "Angle: {}".format("NaN"),
                        (30, 50),
                        (0, 200, 200),
                        (0, 0, 0),
                    )
            self.prep_data(
                [
                    [relative_dataset[0], relative_dataset[1]],
                    self.angle,
                    self.split,
                    self.depth,
                ]
            )
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        vid.release()
        return self.collected_data


if __name__ == "__main__":
    yeti = Yeti()
    yeti.sensor_data_fetch()
    yeti.inference("videos/gp1.mp4")
