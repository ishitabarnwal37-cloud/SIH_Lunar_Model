import math
import numpy as np

R = 1737400.0       # Moon radius, m
RES = 5.687         # working resolution, m/pixel

# Geographic intersection from metadata
lat_max = 0.369695
lat_min = -0.444178

# TMC left boundary changes with latitude
tmc_top_lon = 23.570453
tmc_bottom_lon = 23.548896

# OHRC right boundary
ohrc_top_lon = 23.593739
ohrc_bottom_lon = 23.575386

lat_mid = (lat_max + lat_min) / 2
m_per_deg_lat = math.pi * R / 180
m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat_mid))

height_m = (lat_max - lat_min) * m_per_deg_lat

top_width_m = (ohrc_top_lon - tmc_top_lon) * m_per_deg_lon
bottom_width_m = (ohrc_bottom_lon - tmc_bottom_lon) * m_per_deg_lon

height_px = round(height_m / RES)
top_width_px = round(top_width_m / RES)
bottom_width_px = round(bottom_width_m / RES)

print(f"Common height     : {height_m/1000:.3f} km")
print(f"Common rows       : {height_px}")

print(f"Top overlap width : {top_width_m:.1f} m")
print(f"Top pixels        : {top_width_px}")

print(f"Bottom width      : {bottom_width_m:.1f} m")
print(f"Bottom pixels     : {bottom_width_px}")
