import rasterio
import rasterio.plot

data_name = "data/tif/0landset_bands.tif"
tiff = rasterio.open(data_name)
rasterio.plot.show(tiff)