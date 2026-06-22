# Cell-Crops
crop.py: Given a composite image with cell boundaries outlined in yellow and a table containing one row per cell with nucleus-center coordinates, this function identifies the target cell surrounding each nucleus, masks all pixels outside its boundary to black, and retains only the target cell’s image content.

crop2.py: Generates and saves per-gene grids of representative single-cell crops across multiple imaging channels, including a composite image and an isolated cell-outline view.
