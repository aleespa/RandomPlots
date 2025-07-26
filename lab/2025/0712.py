import os
import subprocess
from pathlib import Path

from common.image_processing import create_directory
from common.technology import images_to_video

OUTPUT_PATH = Path('outputs')

def main():

    width = 3840
    height = 2160
    x_center = -1
    y_center = 0
    zoom = 2.3
    max_iterations = 1000
    n = 100
    for i in range(1, 400):
        output_filename = OUTPUT_PATH / "Mandelbrot" / f'frame{str(i).zfill(4)}.png'
        create_directory(output_filename.parent)
        r = subprocess.run(
            [
                r"C:\Users\Alejandro\Projects\CppFractals\x64\Release\CppFractals.exe",
                str(width),
                str(height),
                str(int(i)),
                str(x_center),
                str(y_center),
                str(zoom),
                str(output_filename.absolute()),
            ],
        )

    print(os.environ['PATH'])
    images_to_video(OUTPUT_PATH / "Mandelbrot", "mandelbrot.mp4", 30)

    import sys

    print(sys.executable)

if __name__ =="__main__":
    main()
