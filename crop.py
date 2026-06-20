def plot_nuclei_crops(df, r, gspath):
    import io
    import subprocess
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.ndimage import (
        binary_closing,
        binary_dilation,
        label
    )

    df1 = df.sort_values("PC1").reset_index(drop=True)

    gspath = gspath.rstrip("/")
    well = gspath.split("/")[-1]

    image_cache = {}

    def format_site(site):
        return str(int(site))

    def load_image(site):
        site = format_site(site)

        if site not in image_cache:
            image_path = (
                f"{gspath}/"
                f"Project_{well}_{site}_3_CompositeLP.png"
            )

            image_bytes = subprocess.run(
                ["gcloud", "storage", "cat", image_path],
                check=True,
                stdout=subprocess.PIPE
            ).stdout

            image_cache[site] = np.asarray(
                Image.open(io.BytesIO(image_bytes)).convert("RGB")
            )

        return image_cache[site]

    def crop_around_center(image, center_x, center_y):
        image_height, image_width = image.shape[:2]
        crop_size = 2 * r + 1

        crop = np.zeros(
            (crop_size, crop_size, 3),
            dtype=image.dtype
        )

        x1 = center_x - r
        x2 = center_x + r + 1
        y1 = center_y - r
        y2 = center_y + r + 1

        image_x1 = max(x1, 0)
        image_x2 = min(x2, image_width)
        image_y1 = max(y1, 0)
        image_y2 = min(y2, image_height)

        crop_x1 = image_x1 - x1
        crop_x2 = crop_x1 + (image_x2 - image_x1)
        crop_y1 = image_y1 - y1
        crop_y2 = crop_y1 + (image_y2 - image_y1)

        crop[crop_y1:crop_y2, crop_x1:crop_x2] = image[
            image_y1:image_y2,
            image_x1:image_x2
        ]

        return crop

    def mask_outside_cell(crop):
        yellow = (
            (crop[:, :, 0] >= 245) &
            (crop[:, :, 1] >= 245) &
            (crop[:, :, 2] <= 20)
        )

        four_connected = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ],
            dtype=bool
        )

        barrier = binary_closing(
            yellow,
            structure=np.ones((3, 3), dtype=bool),
            iterations=1
        )

        barrier = binary_dilation(
            barrier,
            structure=four_connected,
            iterations=1
        )

        # Treat the crop edges as yellow barriers
        barrier[0, :] = True
        barrier[-1, :] = True
        barrier[:, 0] = True
        barrier[:, -1] = True

        traversable = ~barrier

        labeled_regions, _ = label(
            traversable,
            structure=four_connected
        )

        center_y = crop.shape[0] // 2
        center_x = crop.shape[1] // 2

        available = np.argwhere(traversable)

        nearest_index = np.argmin(
            (available[:, 0] - center_y) ** 2 +
            (available[:, 1] - center_x) ** 2
        )

        seed_y, seed_x = available[nearest_index]
        target_region = labeled_regions[seed_y, seed_x]

        inside = labeled_regions == target_region

        target_outline = yellow & binary_dilation(
            inside,
            structure=np.ones((3, 3), dtype=bool),
            iterations=2
        )

        keep = inside | target_outline

        masked_crop = np.zeros_like(crop)
        masked_crop[keep] = crop[keep]

        return masked_crop

    for start in range(0, len(df1), 100):
        chunk = df1.iloc[start:start + 100]

        fig, axes = plt.subplots(
            10,
            10,
            figsize=(15, 15)
        )

        axes = axes.ravel()

        for ax, (_, row) in zip(axes, chunk.iterrows()):
            image = load_image(row["Metadata_Site"])

            center_x = int(
                round(row["Metadata_Nuclei_Location_Center_X"])
            )

            center_y = int(
                round(row["Metadata_Nuclei_Location_Center_Y"])
            )

            crop = crop_around_center(
                image,
                center_x,
                center_y
            )

            masked_crop = mask_outside_cell(crop)

            ax.imshow(masked_crop)
            ax.axis("off")

        for ax in axes[len(chunk):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
