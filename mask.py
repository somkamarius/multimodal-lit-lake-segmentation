import os

import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import unary_union

import numpy as np
import matplotlib.pyplot as plt


def poly_from_utm(polygon, transform):
    poly_pts = []

    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        # Convert polygons to the image CRS
        poly_pts.append(~transform * tuple(i))

    # Generate a polygon object
    new_poly = Polygon(poly_pts)
    return new_poly


train_df = gpd.read_file("ezerai/lt_data/357_Ezerai_tvenkiniai_polygon_proj.shp")

for i in range(340, 357):
    raster_path = "data/images/" + str(i) + ".tif"
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta

    # print("CRS Raster: {}, CRS Vector {}".format(train_df.crs, src.crs))


    # Generate polygon

    # Generate Binary maks

    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    print(poly_shp, type(im_size))
    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size)
    # print(mask)

    # Plot the mask

    plt.figure(figsize=(15, 15))
    plt.imshow(mask)

    mask = mask.astype("uint16")
    save_path = "data/masks/" + str(i) + ".tif"
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    with rasterio.open(save_path, 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)
