import torch
import cv2
import numpy as np
import os

import segmentation_refinement as refine

from mfnet_spec import MFNetModified


model = MFNetModified(
    rgb_ch=MFNetModified.DEFAULT_RGB_CH_SIZE,
    inf_ch=MFNetModified.DEFAULT_INF_CH_SIZE,
    n_class=5,
)
model.load_state_dict(
    torch.load(
        "weights/model_23_04_25_12_50_58_epoch100.pt",
        map_location=torch.device("cuda"),
    ),
)
model.eval()

BACKGROUND_COLORMAP_VALUE = cv2.applyColorMap(
    np.array([0, 0, 0]).astype(np.uint8), cv2.COLORMAP_PARULA
)[0][0]

refiner = refine.Refiner(device="cuda")  #'cuda:0')


def eval(rgb_image, thermal_image):
    combined_image = np.dstack((rgb_image, np.atleast_3d(thermal_image))).astype(
        np.float32
    )

    norm_image = combined_image / 255.0
    norm_image = torch.tensor(
        np.transpose(np.stack([norm_image], axis=0), (0, 3, 1, 2))
    )

    with torch.no_grad():
        logits = model(norm_image)
        predictions = logits.argmax(1)
        predictions = predictions.permute(1, 2, 0).numpy().astype(np.uint8)

        car_refined = refiner.refine(
            combined_image[:, :, 1:].astype(np.uint8),
            (predictions == 1).astype(np.uint8) * 255,
            fast=False,
            L=900,
        )
        person_refined = refiner.refine(
            combined_image[:, :, 1:].astype(np.uint8),
            (predictions == 2).astype(np.uint8) * 255,
            fast=False,
            L=900,
        )

        # cv2.imshow("ref", (predictions == 1).astype(np.uint8) * 255)
        # cv2.imshow("refined", car_refined)

        car_refined = (car_refined > 128).astype(np.uint8)
        person_refined =  (person_refined > 128).astype(np.uint8) * 2

        print(np.max(car_refined), np.max(person_refined))

        predictions = np.sum(
            (car_refined, person_refined),
            axis=0,
        )

    return predictions


def main():
    i = 995

    # os.mkdir("datacapture/data/collection1/out/")

    while i < 3400:
        # Load images
        rgb_image = cv2.imread(f"datacapture/data/collection1/rgb/{i:06d}.jpg")
        thermal_image = cv2.imread(
            f"datacapture/data/collection1/flir/{i:06d}.jpg", cv2.IMREAD_GRAYSCALE
        )

        # Evaluate
        predictions = eval(rgb_image, thermal_image)

        # Visualize
        predictions_vis = predictions * 255 / 5
        predictions_vis = cv2.applyColorMap(
            predictions_vis.astype(np.uint8), cv2.COLORMAP_PARULA
        )
        predictions_vis[predictions_vis[:, :, 0] == BACKGROUND_COLORMAP_VALUE[0]] = 0
        predictions_vis[predictions_vis[:, :, 1] == BACKGROUND_COLORMAP_VALUE[1]] = 0
        predictions_vis[predictions_vis[:, :, 2] == BACKGROUND_COLORMAP_VALUE[2]] = 0

        # Overlay on RGB image
        predictions_vis = cv2.addWeighted(rgb_image, 0.5, predictions_vis, 0.5, 0)
        cv2.imwrite(f"datacapture/data/collection1/out/{i:06d}.jpg", predictions_vis)
        cv2.imshow("Output", predictions_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        print(i)
        i += 1


if __name__ == "__main__":
    main()
