# Disable TensorFlow info messages
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import cv2
import tensorflow as tf
import PIL


class Upscaler:
    def __init__(self, upscaler_model, denoiser_model):
        self.upscaler = tf.keras.models.load_model(upscaler_model)
        self.denoiser = tf.keras.models.load_model(
            denoiser_model,
            custom_objects={
                "rgb_to_ycbcr": self._rgb_to_ycbcr,
                "ycbcr_to_rgb": self._ycbcr_to_rgb,
            },
        )

    def _rgb_to_ycbcr(self, image):
        return tf.image.rgb_to_yuv(image)

    def _ycbcr_to_rgb(self, y_pred, image):
        y = tf.clip_by_value(y_pred, 0, 1)
        u = image[..., 1:2]
        v = image[..., 2:3]
        yuv = tf.concat([y, u, v], axis=-1)
        return tf.image.yuv_to_rgb(yuv)

    def _pad_image(self, image, target_size=(256, 256)):
        width, height = image.size
        padded_image = PIL.Image.new("RGB", target_size, color=(0, 0, 0))
        padded_image.paste(
            image, ((target_size[0] - width) // 2, (target_size[1] - height) // 2)
        )
        return padded_image

    def _denoise_tile(self, image):
        original_size = image.size
        padded_image = self._pad_image(image)

        denoised = self.denoiser.predict(
            np.expand_dims(tf.keras.utils.img_to_array(padded_image) / 255.0, axis=0),
            verbose=0,
        )
        denoised = denoised.clip(0, 1)

        denoised_image = PIL.Image.fromarray((denoised[0] * 255.0).astype(np.uint8))

        width, height = original_size
        left = (denoised_image.width - width) // 2
        top = (denoised_image.height - height) // 2
        right = left + width
        bottom = top + height
        cropped_image = denoised_image.crop((left, top, right, bottom))

        return cropped_image

    def _upscale_tile(self, img):
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()

        y = tf.keras.utils.img_to_array(y)
        y = y.astype("float32") / 255.0

        input = np.expand_dims(y, axis=0)
        out = self.upscaler.predict(input, verbose=0)

        out_img_y = out[0]
        out_img_y = out_img_y.clip(0, 1)
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
        out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
        out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
            "RGB"
        )
        return out_img

    def upscale_image(self, file, denoise):
        image = cv2.imread(file)

        tile_size = 128
        overlap = 16
        h, w, _ = image.shape
        scaled_h, scaled_w = h * 2, w * 2

        upscaled_image = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                tile = image[y : y + tile_size, x : x + tile_size]

                upscaled_tile = self._upscale_tile(PIL.Image.fromarray(tile))

                if denoise:
                    upscaled_tile = self._denoise_tile(upscaled_tile)

                result_y = y * 2
                result_x = x * 2
                tile_w, tile_h = upscaled_tile.size

                alpha = np.ones((tile_h, tile_w, 1))
                if y != 0:
                    alpha[: overlap * 2, :, 0] = np.linspace(0, 1, overlap * 2).reshape(
                        -1, 1
                    )
                if x != 0:
                    alpha[:, : overlap * 2, 0] = np.linspace(0, 1, overlap * 2).reshape(
                        1, -1
                    )

                upscaled_image[
                    result_y : result_y + tile_h, result_x : result_x + tile_w
                ] = (
                    alpha * upscaled_tile
                    + (1 - alpha)
                    * upscaled_image[
                        result_y : result_y + tile_h, result_x : result_x + tile_w
                    ]
                ).astype(
                    np.uint8
                )

        return upscaled_image
