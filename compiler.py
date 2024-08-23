import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import RectBivariateSpline

#files = [f for f in os.listdir('./balls') if f.endswith('.txt')]
#print(files)

#data = []
#for file in files:
#	with open('./balls/'+ file) as f:
#		s = f.read()
#		data.append(s.split())

data = pd.read_csv('circle_coordinates.csv', header=None)

#data.iloc[8][-20:] += 990 - data.iloc[8][-23:]
#data.iloc[8][:51] -= 2050
#data = [[float(item) for item in row] for row in data]
#normalized_list = [[item - min(row) for item in row] for row in data]

row_min = data.min(axis=1)
data = data.sub(row_min, axis=0)
#data.iloc[9][:69] -= 120

# Create a grid of coordinates for the smooth plot
#y_smooth = np.linspace(0, y[-1], 100)
#x_smooth, y_smooth = np.meshgrid(x, y_smooth)
#z_smooth = spl(y_smooth)

z = data.values
y = data.columns.values
x = data.index.values
print(x.shape,y.shape,z.shape)

spline = RectBivariateSpline(x, y, z, s = 500000)

z_smooth = spline(x, y)

# Create the 3D surface plot
fig = go.Figure(data=[go.Surface(z=z_smooth, x=y, y=x)])

fig.update_layout(scene=dict(
                    xaxis_title='Y Axis',
                    yaxis_title='X Axis',
                    zaxis_title='Z Axis'),
                    title='3D Scatter Plot')
fig.show()
