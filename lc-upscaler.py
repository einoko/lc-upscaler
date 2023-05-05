import click
import os
import cv2
import sys
from upscale import Upscaler

# Model paths
upscaler_path = "./models/upscaler.h5"
denoiser_path = "./models/denoiser.h5"


@click.command()
@click.option(
    "--scale", default=2.00, help="Ratio to upscale the image (between 1.00 and 2.00)"
)
@click.option("--compress", is_flag=True, help="Compress image using JPEG compression")
@click.option(
    "--denoise",
    is_flag=True,
    help="Denoise image using a neural network (experimental, slow)",
)
@click.option(
    "--quality",
    default=90,
    help="Quality of the compressed output image (between 1 and 100, default: 90)",
)
@click.option(
    "--bw",
    is_flag=True,
    help="Save image as grayscale (for black and white pages)",
)
@click.option("--suffix", default="_upscaled", help="Suffix added to the output file")
@click.argument("file")
def main(
    scale: float,
    compress: bool,
    denoise: bool,
    quality: int,
    bw: bool,
    suffix: str,
    file: str,
):
    if not os.path.isfile(file):
        print(f"Error: File '{file}' does not exist. Check the path.")
        sys.exit(1)

    if scale < 1.00 or scale > 2.00:
        print("Error: Ratio must be between 1.00 and 2.00")
        sys.exit(1)

    filename = file.split(".")[0]
    extension = "jpg" if compress else "png"

    print(f"Converting '{file}' to '{filename}{suffix}.{extension}'")

    try:
        upscaler = Upscaler(upscaler_path, denoiser_path)
        upscaled_image = upscaler.upscale_image(file, denoise)

        if scale != 2.00:
            upscaled_image = cv2.resize(
                upscaled_image,
                (0, 0),
                fx=scale / 2.00,
                fy=scale / 2.00,
                interpolation=cv2.INTER_AREA,
            )

        if bw:
            upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2GRAY)

        if compress:
            cv2.imwrite(
                f"{filename}{suffix}.{extension}",
                upscaled_image,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
        else:
            cv2.imwrite(f"{file.split('.')[0]}{suffix}.{extension}", upscaled_image)

        print("Done!")
    except Exception as e:
        print(e)
        print("Error: Could not upscale image")


if __name__ == "__main__":
    main()
