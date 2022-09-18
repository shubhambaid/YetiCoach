from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from evaluate_depth import STEREO_SCALE_FACTOR


class MonoDepth:
    def __init__(self):
        model_path = "models/mono_1024x320"
        encoder_path = model_path + "/encoder.pth"
        depth_decoder_path = model_path + "/depth.pth"

        print("[Loading Pretrained Encoder]")
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location="cuda")

        self.feed_height = loaded_dict_enc["height"]
        self.feed_width = loaded_dict_enc["width"]
        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to("cuda")
        self.encoder.eval()

        print("[Loading Pretrained Decoder]")
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4)
        )
        loaded_dict = torch.load(depth_decoder_path, map_location="cuda")
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to("cuda")
        self.depth_decoder.eval()

    def convert_pil_npy(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = pil.fromarray(img)
        # im_np = np.asarray(im_pil)
        return pil_img

    def calculate_approx_depth(self, image, coords):
        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
        cropped = image[y1-int(y1/2):y2-int(y2/2), x1:x2]
        average_color_row = np.average(cropped, axis=0)
        average_color = np.average(average_color_row, axis=0)
        return average_color

    def get_depth(self, image, coords):
        with torch.no_grad():
            input_image = self.convert_pil_npy(image)
            original_width, original_height = input_image.size
            input_image = input_image.resize(
                (self.feed_width, self.feed_height), pil.LANCZOS
            )
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            input_image = input_image.to("cuda")
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp,
                (original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )

            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(
                np.uint8
            )
            colormapped_im = cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR)
            return (colormapped_im, self.calculate_approx_depth(colormapped_im, coords))


if __name__ == "__main__":
    img = cv2.imread("dataset/gp3_115.jpeg")
    mono = MonoDepth()
    print(mono.get_depth(img, [630, 443, 835, 697]))
    cv2.imshow("test", img)
    cv2.waitKey(0)
