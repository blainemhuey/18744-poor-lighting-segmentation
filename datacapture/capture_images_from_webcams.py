import os
import json

import cv2
import numpy as np

STORE_DATA = True

FRAMES_TO_SKIP = 10

FLIR_CAMERA = 1
RGB_CAMERA = 0

DATA_FOLDER = "data"
CALIB_FILE = "camera_calib.json"

RGB_OFFSET = (-12, -10)
FLIR_CROP = 20


def main():
    frame_number = 0
    skip_counter = FRAMES_TO_SKIP

    # Load camera calibration
    with open(CALIB_FILE, "r") as f:
        calib = json.load(f)
    flir_cameraMatrix = np.array(calib["flir"]["cameraMatrix"])
    flir_distortionCoefficients = np.array(calib["flir"]["distortionCoefficients"])
    rgb_cameraMatrix = np.array(calib["rgb"]["cameraMatrix"])
    rgb_distortionCoefficients = np.array(calib["rgb"]["distortionCoefficients"])

    # Create background image of a checkerboard pattern
    checkerboard = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(0, 480, 10):
        for j in range(0, 640, 10):
            checkerboard[i : i + 5, j : j + 5, 2] = 255
            checkerboard[i + 5 : i + 10, j + 5 : j + 10, 2] = 255
    checkerboard_normalized = checkerboard.astype(np.float32) / np.max(checkerboard)

    # Open cameras
    flir = cv2.VideoCapture(FLIR_CAMERA)
    flir.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    flir.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    rgb = cv2.VideoCapture(RGB_CAMERA)
    rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    # Create data folder
    if STORE_DATA and not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        os.makedirs(os.path.join(DATA_FOLDER, "flir"))
        os.makedirs(os.path.join(DATA_FOLDER, "rgb"))

    while True:
        # Read frames
        _, flir_frame = flir.read()
        _, rgb_frame = rgb.read()

        # # Make flir frame single-channel
        flir_frame = cv2.cvtColor(flir_frame, cv2.COLOR_BGR2GRAY)

        # Resize frames
        flir_frame = cv2.resize(flir_frame, (640, 480))
        rgb_frame = cv2.resize(rgb_frame, (910, 480))

        # Undistort frames
        flir_frame = cv2.undistort(
            flir_frame,
            flir_cameraMatrix,
            flir_distortionCoefficients,
        )
        rgb_frame = cv2.undistort(
            rgb_frame,
            rgb_cameraMatrix,
            rgb_distortionCoefficients,
        )

        # Crop center 640 pixels of rgb frame
        rgb_frame = rgb_frame[:, 135:775]

        # Crop in on the flir frame
        flir_frame = flir_frame[
            FLIR_CROP : flir_frame.shape[0] - FLIR_CROP,
            FLIR_CROP : flir_frame.shape[1] - FLIR_CROP,
        ]

        # Shift the RGB frame to align with the FLIR frame
        rgb_frame = np.roll(rgb_frame, RGB_OFFSET[0], axis=0)
        rgb_frame = np.roll(rgb_frame, RGB_OFFSET[1], axis=1)

        # Crop in on both frame the amount of the offset, then resize back to original
        rgb_frame = rgb_frame[
            abs(RGB_OFFSET[0]) : rgb_frame.shape[0] - abs(RGB_OFFSET[0]),
            abs(RGB_OFFSET[1]) : rgb_frame.shape[1] - abs(RGB_OFFSET[1]),
            :,
        ]
        flir_frame = flir_frame[
            abs(RGB_OFFSET[0]) : flir_frame.shape[0] - abs(RGB_OFFSET[0]),
            abs(RGB_OFFSET[1]) : flir_frame.shape[1] - abs(RGB_OFFSET[1]),
        ]
        rgb_frame = cv2.resize(rgb_frame, (640, 480))
        flir_frame = cv2.resize(flir_frame, (640, 480))

        # Alpha-blend checkerboard pattern over rgb frame using the flir frame as an alpha mask
        rgb_normalized = rgb_frame.astype(np.float32) / np.max(rgb_frame)
        flir_normalized = flir_frame[:, :, np.newaxis].astype(np.float32)
        flir_normalized = np.square(flir_normalized - np.average(flir_normalized))
        flir_normalized = flir_normalized / np.max(flir_normalized)

        combined_frame = checkerboard_normalized * flir_normalized + rgb_normalized * (
            1 - flir_normalized
        )

        # Normalize combined frame
        combined_frame = combined_frame * 255 / np.max(combined_frame)

        # Canny-edge the rgb frame, draw edges in yellow on the combined frame
        rgb_blur = cv2.GaussianBlur(rgb_frame, (0, 0), 2)
        rgb_edges = cv2.Canny(rgb_blur, 7, 35)
        rgb_edges = cv2.dilate(
            rgb_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        combined_frame[rgb_edges > 0, 1] = 255
        combined_frame[rgb_edges > 0, 2] = 255

        # Canny-edge the flir frame, draw edges in cyan on the combined frame
        flir_blur = cv2.GaussianBlur(flir_frame, (0, 0), 2)
        flir_edges = cv2.Canny(flir_blur, 2, 15)
        flir_edges = cv2.dilate(
            flir_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        combined_frame[flir_edges > 0, 0] = 255
        combined_frame[flir_edges > 0, 1] = 255

        # Display frame
        cv2.imshow(
            "Cameras",
            np.hstack(
                (
                    cv2.cvtColor(flir_frame, cv2.COLOR_GRAY2BGR),
                    rgb_frame,
                    combined_frame.astype(np.uint8),
                )
            ),
        )

        if skip_counter == 0:
            skip_counter = FRAMES_TO_SKIP

            # Save frames
            if STORE_DATA:
                cv2.imwrite(
                    os.path.join(DATA_FOLDER, "flir", f"{frame_number:06d}.jpg"),
                    flir_frame,
                )
                cv2.imwrite(
                    os.path.join(DATA_FOLDER, "rgb", f"{frame_number:06d}.jpg"),
                    rgb_frame,
                )
            frame_number += 1
        else:
            skip_counter -= 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
