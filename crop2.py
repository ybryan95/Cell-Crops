def save_representative_gene_crops(
    rep,
    box_size,
    output_dir="gs://outputDir"
):
    from scipy.ndimage import (
        binary_closing,
        binary_dilation,
        convolve,
        label
    )

    channels = ["AGP", "DNA", "LowBODIPYRNA", "Mito"]

    base_gcs_path = (
        "gs://projectDir"
        "imgDIr"
    )

    outline_base_path = (
        "gs://projectDir"
        "imgDIr_output2"
    )

    fs = gcsfs.GCSFileSystem()
    image_cache = {}
    outline_cache = {}
    output_files = []

    half_box_size = box_size // 2

    def read_image(image_path):
        if image_path not in image_cache:
            with fs.open(image_path, "rb") as file:
                image = np.squeeze(tifffile.imread(file))

            image = exposure.rescale_intensity(
                image,
                in_range=(
                    image.min(),
                    np.percentile(image, 99.95)
                ),
                out_range=(0, 1)
            ).astype(np.float32)

            image_cache[image_path] = image

        return image_cache[image_path]

    def read_outline_image(image_path):
        if image_path not in outline_cache:
            with fs.open(image_path, "rb") as file:
                image = np.squeeze(tifffile.imread(file))

            image = exposure.rescale_intensity(
                image,
                in_range=(image.min(), image.max()),
                out_range=(0, 1)
            ).astype(np.float32)

            outline_cache[image_path] = image

        return outline_cache[image_path]

    def mask_outside_cell(crop):
        yellow = (
            (crop[:, :, 0] >= 0.96)
            & (crop[:, :, 1] >= 0.96)
            & (crop[:, :, 2] <= 0.08)
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
            (available[:, 0] - center_y) ** 2
            + (available[:, 1] - center_x) ** 2
        )

        seed_y, seed_x = available[nearest_index]
        target_region = labeled_regions[seed_y, seed_x]

        inside = labeled_regions == target_region

        masked_crop = np.zeros_like(crop)
        masked_crop[inside] = crop[inside]

        return masked_crop

    def replace_white_outline(crop):
        result = crop.copy()

        white = np.all(result >= 0.95, axis=2)

        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0

        while white.any():
            valid = (
                (~white)
                & np.any(result > 0, axis=2)
            )

            neighbor_count = convolve(
                valid.astype(np.float32),
                kernel,
                mode="constant",
                cval=0
            )

            fillable = white & (neighbor_count > 0)

            if not fillable.any():
                break

            for channel_index in range(3):
                neighbor_sum = convolve(
                    result[:, :, channel_index]
                    * valid.astype(np.float32),
                    kernel,
                    mode="constant",
                    cval=0
                )

                result[:, :, channel_index][fillable] = (
                    neighbor_sum[fillable]
                    / neighbor_count[fillable]
                )

            white[fillable] = False

        return result

    def format_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("white")
            spine.set_linewidth(1.5)

    for gene, gene_df in rep.groupby("Metadata_Gene_List"):
        gene_df = gene_df.reset_index(drop=True)

        rows_count = gene_df.shape[0]
        columns_count = len(channels) + 2

        fig, axes = plt.subplots(
            rows_count,
            columns_count,
            figsize=(columns_count * 2, rows_count * 2),
            squeeze=False,
            gridspec_kw={
                "wspace": 0,
                "hspace": 0
            }
        )

        fig.patch.set_facecolor("white")

        for row_index, row in gene_df.iterrows():
            well = str(row["Metadata_Well"])
            pseudo_well = str(row["Metadata_PseduoWell"])
            site = int(row["Metadata_Site"])

            x_center = int(
                row["Metadata_Nuclei_Location_Center_X"]
            )

            y_center = int(
                row["Metadata_Nuclei_Location_Center_Y"]
            )

            outline_x_center = int(
                row["Metadata_Nuclei_AreaShape_Center_X"]
            )

            outline_y_center = int(
                row["Metadata_Nuclei_AreaShape_Center_Y"]
            )

            crops = {}

            for channel_index, channel in enumerate(channels):
                image_path = (
                    f"{base_gcs_path}/"
                    f"noFFA_{well}_Corr{channel}_Site_{site}.tiff"
                )

                image = read_image(image_path)

                crop = image[
                    y_center - half_box_size:
                    y_center + half_box_size,
                    x_center - half_box_size:
                    x_center + half_box_size
                ]

                crops[channel] = crop

                ax = axes[row_index, channel_index]

                ax.imshow(
                    crop,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )

                if row_index == 0:
                    ax.set_title(channel, pad=1)

                format_axis(ax)

            composite = np.zeros(
                (
                    crops["DNA"].shape[0],
                    crops["DNA"].shape[1],
                    3
                ),
                dtype=np.float32
            )

            composite[:, :, 0] = np.clip(
                crops["Mito"] + crops["AGP"],
                0,
                1
            )

            composite[:, :, 1] = np.clip(
                crops["LowBODIPYRNA"] + crops["AGP"],
                0,
                1
            )

            composite[:, :, 2] = crops["DNA"]

            composite_ax = axes[row_index, len(channels)]

            composite_ax.imshow(composite)

            if row_index == 0:
                composite_ax.set_title("Composite", pad=1)

            format_axis(composite_ax)

            outline_path = (
                f"{outline_base_path}/{pseudo_well}/"
                f"noFFA_{pseudo_well}_{site}_CompositeLP.tiff"
            )

            outline_image = read_outline_image(outline_path)

            outline_crop = outline_image[
                outline_y_center - half_box_size:
                outline_y_center + half_box_size,
                outline_x_center - half_box_size:
                outline_x_center + half_box_size
            ]

            masked_outline_crop = mask_outside_cell(
                outline_crop
            )

            masked_outline_crop = replace_white_outline(
                masked_outline_crop
            )

            outline_ax = axes[row_index, -1]

            outline_ax.imshow(masked_outline_crop)

            if row_index == 0:
                outline_ax.set_title("w/ outline", pad=1)

            format_axis(outline_ax)

        fig.subplots_adjust(
            left=0,
            right=1,
            top=0.98,
            bottom=0,
            wspace=0,
            hspace=0
        )

        output_path = f"{output_dir}/{gene}.png"

        with fs.open(output_path, "wb") as file:
            fig.savefig(
                file,
                format="png",
                dpi=100,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0
            )

        plt.close(fig)

        output_files.append(output_path)
        print(f"Saved: {output_path}")

    return output_files
