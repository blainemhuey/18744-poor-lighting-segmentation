from PIL import Image
import numpy as np

from pathlib import Path
import multiprocessing


FLIR_PATH = "data/collection1/flir/"
RGB_PATH = "data/collection1/rgb/"
OUTPUT_PATH = "data/collection1/merged/"

def merge_images(flir_file):
    flir_name = flir_file.name
    rgb_file = Path(RGB_PATH) / flir_name
    
    if rgb_file.exists():
        flir_image = Image.open(flir_file)
        rgb_image = Image.open(rgb_file)
        
        flir_image.load()
        rgb_image.load()
        
        flir_array = np.asarray(flir_image)
        rgb_array = np.asarray(rgb_image)

        merged_array = np.concatenate((rgb_array, flir_array[:, :, np.newaxis]), axis=2)
        merged_image = Image.fromarray(merged_array)
        merged_image.save(Path(OUTPUT_PATH) / (flir_file.stem + ".png"), mode="RGBA")

if __name__ == "__main__":
    flir_files = Path(FLIR_PATH).glob("*")
    
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    tp = multiprocessing.Pool(multiprocessing.cpu_count())
    
    tp.map(merge_images, flir_files)

    tp.close()
    
