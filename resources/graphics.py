import cv2
import numpy as np
from typing import Optional, Tuple

class Graphics():
    def __init__(self):
        canvas_size = [800,800] 
        background_image = cv2.imread('./resources/pump_background_.png')
        # get scale to rescale other images
        scale = np.array(canvas_size)/background_image.shape[0:2]
        self.background_image = cv2.resize(background_image, (canvas_size[0], canvas_size[1]))
        
        # get valve images
        self.image_pump_open = self._load_and_resize('./resources/pump_valve_open.png', scale)
        self.image_pump_closed = self._load_and_resize('./resources/pump_valve_closed.png', scale)

    def render_valve_and_motor(self, valve_states, MOTOR_POS):
        display_image = self.background_image
        if valve_states[0] == 1:
            display_image = self._overlay_images(display_image, self.image_pump_open, 166, 25)
        else:
            display_image = self._overlay_images(display_image, self.image_pump_closed, 166, 25)

        if valve_states[1] == 1:
            display_image = self._overlay_images(display_image, self.image_pump_open, 585, 25)
        else:
            display_image = self._overlay_images(display_image, self.image_pump_closed, 585, 25)

        if valve_states[2] == 1:
            display_image = self._overlay_images(display_image, self.image_pump_open, 378, 217)
        else:
            display_image = self._overlay_images(display_image, self.image_pump_closed, 378, 217)

        motor_position =  int(393+MOTOR_POS)
        cv2.line(display_image, (motor_position,25),(motor_position,177), (255, 71, 20), 3)
        self.display_image = display_image
        # return display_image

    def _load_and_resize(self, image_path, scale):
        def _remove_background(image_bgr):
            # get the image dimensions (height, width and channels)
            h, w, c = image_bgr.shape
            # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
            image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
            # create a mask where white pixels ([255, 255, 255]) are True
            white = np.all(image_bgr >= [245, 245, 245], axis=-1)
            # change the values of Alpha to 0 for all the white pixels
            image_bgra[white, -1] = 0
            return image_bgra
            
        image = cv2.imread(image_path)
        image = cv2.resize(image, (int(image.shape[0]*scale[0]), int(image.shape[1]*scale[1])))
        image = _remove_background(image)
        return image

    def _overlay_images(self, background_image, overlay_image, x_offset, y_offset):
        y1, y2 = y_offset, y_offset + overlay_image.shape[0]
        x1, x2 = x_offset, x_offset + overlay_image.shape[1]
        alpha_s = overlay_image[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            background_image[y1:y2, x1:x2, c] = (alpha_s * overlay_image[:, :, c] +
                                    alpha_l * background_image[y1:y2, x1:x2, c])
        return background_image
    

    def add_text_to_image(
        self,
        label: str,
        # image_rgb: np.ndarray,
        top_left_xy: Tuple = (0, 0),
        font_scale: float = 0.5,
        font_thickness: float = 1,
        font_face=cv2.FONT_HERSHEY_COMPLEX,
        font_color_rgb: Tuple = (0, 0, 0),
        bg_color_rgb: Optional[Tuple] = None,
        outline_color_rgb: Optional[Tuple] = None,
        line_spacing: float = 1,
    ):
        """
        This is necessary because opencv doesn't support newline...
        Adds text (including multi line text) to images.
        You can also control background color, outline color, and line spacing.

        outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
        """
        image_rgb = self.display_image
        OUTLINE_FONT_THICKNESS = 3 * font_thickness
        im_h, im_w = image_rgb.shape[:2]

        for line in label.splitlines():
            x, y = top_left_xy

            # ====== get text size
            if outline_color_rgb is None:
                get_text_size_font_thickness = font_thickness
            else:
                get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

            (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
                line,
                font_face,
                font_scale,
                get_text_size_font_thickness,
            )
            line_height = line_height_no_baseline + baseline

            if bg_color_rgb is not None and line:
                # === get actual mask sizes with regard to image crop
                if im_h - (y + line_height) <= 0:
                    sz_h = max(im_h - y, 0)
                else:
                    sz_h = line_height

                if im_w - (x + line_width) <= 0:
                    sz_w = max(im_w - x, 0)
                else:
                    sz_w = line_width

                # ==== add mask to image
                if sz_h > 0 and sz_w > 0:
                    bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                    bg_mask[:, :] = np.array(bg_color_rgb)
                    image_rgb[
                        y : y + sz_h,
                        x : x + sz_w,
                    ] = bg_mask

            # === add outline text to image
            if outline_color_rgb is not None:
                image_rgb = cv2.putText(
                    image_rgb,
                    line,
                    (x, y + line_height_no_baseline),  # putText start bottom-left
                    font_face,
                    font_scale,
                    outline_color_rgb,
                    OUTLINE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )
            # === add text to image
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                font_color_rgb,
                font_thickness,
                cv2.LINE_AA,
            )
            top_left_xy = (x, y + int(line_height * line_spacing))

        return image_rgb

# test code
if __name__ == '__main__':
    pass
    # graphics = Graphics()
    # graphics.render([1,0,1])
    # cv2.imshow('a', graphics.display_image)
    # cv2.waitKey()