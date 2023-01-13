import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio.warp
from pyproj import Transformer
from rasterio import CRS
from shapely.geometry import Polygon


shapefile = gpd.read_file("ezerai/lt_data/357_Ezerai_tvenkiniai_polygon_proj.shp")

i = 15

bounds = list(shapefile['geometry'].at[i].bounds)
# print(shapefile.crs)
bounds[2] = bounds[2] + 0.01
bounds[0] = bounds[0] - 0.01
bounds[3] = bounds[3] + 0.01
bounds[1] = bounds[1] - 0.01
bounds_changed = tuple(bounds)

# Let's take a copy of our layer
data_proj = shapefile['geometry'].copy()
# print(data_proj[i])
polygon_geom = Polygon(data_proj[i])
# print(polygon_geom)


# Reproject the geometries by replacing the values with projected ones
# print(data_proj[i].bounds)
# print(type(data_proj), data_proj.crs)
data_proj_32633 = gpd.GeoSeries(data_proj.to_crs(epsg=32633)[i])
print(data_proj_32633.crs)
data_proj_32633.plot(facecolor='black')
plt.axis('off')

# print(bounds)

transformer = Transformer.from_crs(4326, 32633)

xmin, ymin = transformer.transform(bounds_changed[0], bounds_changed[1])
xmax, ymax = transformer.transform(bounds_changed[2], bounds_changed[3])

# print(xmin, ymin, xmax, ymax)
bounds = data_proj.at[i].bounds
print(bounds)
x1, y1 = transformer.transform(bounds_changed[0], bounds_changed[1])
x2, y2 = transformer.transform(bounds_changed[2], bounds_changed[3])
bounds = tuple([x1, y1, x2, y2])
print(bounds)
# maxx = bounds_changed[2] # cia pakeiciam ne tuo crs, negerai.
# # Reiks padidint boundsus jau pačiam faile ir tada convertint į kitą crs
# minx = bounds_changed[0]
# maxy = bounds_changed[3]
# miny = bounds_changed[1]
# print('nu ble', [data_proj_32633.at[i].bounds[0], data_proj_32633.at[i].bounds[2]])
plt.xlim(data_proj_32633.at[0].bounds[0], data_proj_32633.at[0].bounds[2])
plt.ylim(data_proj_32633.at[0].bounds[1], data_proj_32633.at[0].bounds[3])

# print(data_proj_32633.at[i].bounds[3], data_proj_32633.at[i])

# Setting the background color of the plot
# using set_facecolor() method
# ax.set_facecolor("yellow")

# Remove empty white space around the plot
# plt.tight_layout()

plt.savefig(str(i) + '_mask.png', bbox_inches='tight', pad_inches=0, dpi=72)
plt.show()
