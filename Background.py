import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


INPUT_IMAGE_PATH = Path("data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg")
MASK_IMAGE_PATH = Path("output/haar_step1/foreground_mask_visual_adjusted.png") # Step 1 输出的蒙版
OUTPUT_DIRECTORY = Path("output/step2_bokeh")


# todo blur size can be changed in GUI in the end
def apply_bokeh_effect(original_image, mask, blur_kernel_size: int = 31):

    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    blurred_background = cv2.GaussianBlur(original_image, (blur_kernel_size, blur_kernel_size), 0)

    alpha = mask.astype(np.float32) / 255.0

    alpha_3d = np.dstack([alpha, alpha, alpha])

    # Result = Foreground * alpha + Background * (1 - alpha)
    original_float = original_image.astype(np.float32)
    background_float = blurred_background.astype(np.float32)

    portrait_float = (original_float * alpha_3d) + (background_float * (1.0 - alpha_3d))

    portrait_final = np.clip(portrait_float, 0, 255).astype(np.uint8)

    return portrait_final


def main():
    image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read original image: {INPUT_IMAGE_PATH}")

    mask = cv2.imread(str(MASK_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {MASK_IMAGE_PATH}. Please run Step 1 first.")

    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    bokeh_image = apply_bokeh_effect(image, mask, blur_kernel_size=45)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Original Image")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("2. Foreground Mask (From Step 1)")

    axes[2].imshow(cv2.cvtColor(bokeh_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("3. Final Portrait Mode (Step 2)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIRECTORY / "portrait_result.jpg"
    cv2.imwrite(str(out_path), bokeh_image)
    print(f"Success! Portrait mode image saved to {out_path}")


if __name__ == "__main__":
    main()
