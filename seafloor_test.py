from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.mlab import griddata
import scipy.interpolate as interpolate
import sys
import random
import math
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.interpolate import SmoothBivariateSpline
from scipy.interpolate import griddata


from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import box

import ast

def find_nearest(array,x, y, test):
	min_dist = np.inf;
	idx = -1;
	dists = []
	for i,a in enumerate(array):
		v = ( ( y[a[0]]-y[test[0]])**2+(x[a[1]]-x[test[1]])**2 )
		dists.append( [ i, 1.0/math.sqrt(v) ] );
	dists.sort(reverse=True, key=lambda x:x[1]);	
	return np.array( dists );
			

def process_poly(poly):
	lons, lats = poly.exterior.coords.xy
	x,y=(lons, lats);
	pols = list(zip(x,y))
	ipols = [];
	for i in poly.interiors:
		lons, lats = i.coords.xy;
		x,y=(lons, lats);
		ipols.append( list(zip(x,y)) )
			
	return pols, ipols
	
#######################################
##Grid of where the flipping coral is##
#######################################
my_buffer = 25
polygons = []
polygon_coords = []
with open('/local/home/rudyarthur/coral/coral_polygons.dat', 'r') as infile:
	for line in infile:
		p = ast.literal_eval( line );
		poly = Polygon( p ).buffer(my_buffer);
		polygons.append( poly );
		
		pols, ipols = process_poly( poly );
		polygon_coords.append( pols );

coral = [];
with open('/local/home/rudyarthur/coral/PS_bathy_no_coral.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
	for row in spamreader:	
		if row[2] == '1.70E+38':
			x = float(row[0]);
			y = float(row[1]);
			if x > 451920 and x < 454520 and y > 7885390 and y < 7888970:
				if y < 7885670 and x < 452182: continue;
				coral.append( (row[0],row[1]) );	
				
x = [];
y = [];
z = [];
mk = [];
ht = [];
#################################################################
## The CSV grid is not exactly uniform, but it is close enough ##
## Record heights and coral presence
#################################################################
with open('/local/home/rudyarthur/coral/PS_bathy.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
	line = [];
	cline = [];
	xlast = np.inf;
	ylast = np.inf;
	
	gotx = False;
	goty = False;
	
	for row in spamreader:
		
		xcoord = float(row[0]);
		ycoord = float(row[1]);
		
		if row[2] == '1.70E+38':
			zcoord = 0; #1663.0745561; #max height in data set
		else:
			zcoord = float(row[2])*1000;
		
		if (row[0],row[1]) in coral:
			iscoral = 1.0;
		else:
			iscoral = 0.0;
		
		if len(line) == 0:
			y.append( ycoord );
			
		if len(line) > 0 and xcoord < xlast:
			ht.append( line );
			mk.append( cline );
			line = [ zcoord ];
			cline = [ iscoral ];
			gotx = True;
			y.append( ycoord );
			
		else:
			line.append( zcoord );
			cline.append( iscoral );
			if not gotx: x.append( xcoord );
			
		xlast = xcoord
		ylast = ycoord
		
ht.append( line )
mk.append( cline )

##Convert to np arrays
ht = np.array( ht );
mk = np.array( mk );
x = np.array(x)
y = np.array(y)

edges = [ [] for p in polygons ];
edge_map = []
mj = len(x)
mi = len(y)
for i in range( mi ):
	print("edges", mi-i)
	tmp = []
	for j in range( mj ):
		inp = False;
		
		if i>0 and j>0 and i<mi-1 and j<mj-1:
			for r,p in enumerate(polygons):
				pt = Point(x[j], y[i]);
				if p.contains(pt): break;
				
				ptu = Point(x[j], y[i+1]);
				ptd = Point(x[j], y[i-1]);
				ptl = Point(x[j-1], y[i]);
				ptr = Point(x[j+1], y[i]);
				if (p.contains(ptu) or p.contains(ptd) or p.contains(ptl) or p.contains(ptr)):
					if not inp: tmp.append( 2000 );
					edges[r].append( (i,j) );
					inp = True;
					#break;
				
		if not inp:
			tmp.append( 0 );
	edge_map.append(tmp)
edge_map = np.array( edge_map )
#CS = plt.contourf(x,y,edge_map, 60, cmap=plt.cm.ocean_r)
#plt.colorbar()  
#plt.show()
#plt.close();
#sys.exit(1);	

seafloor = [];
nearest = [1,2,4,6,8,10,20,50,100,150,200,250,500];
mj = len(x)
mi = len(y)
for i in range( mi ):
	print("floor", mi-i)
	tmp = [];
	for j in range( mj ):
		est = []
		inp = False;
		pt = Point(x[j], y[i])

		for r,p in enumerate(polygons):
			
			if p.contains( pt ):
 
				weights = find_nearest(edges[r], x, y ,( i,j ) );
				for n in nearest:
					av = 0;
					norm = 0;
					for k in range( min(n, len(weights)) ):
						idx = int(weights[k][0]);
						av += weights[k][1] * ( ht[ edges[r][idx][0] ][ edges[r][idx][1] ] );
						norm += weights[k][1];
					av = av/norm;
					est.append( av );
				inp = True;
				break;

		if not inp:
			for n in nearest:
				est.append( ht[i][j] );
		
		tmp.append( est )
	seafloor.append(tmp)
seafloor = np.array( seafloor );		


for k in range(len(nearest)):
	print("plot", nearest[k])
	CS = plt.contourf(x,y,seafloor[:,:,k], 60, cmap=plt.cm.ocean_r)
	plt.colorbar()  
	#plt.show()
	plt.savefig("seafloor_extrap_" + str(my_buffer) + "_" + str(nearest[k]) + ".png");
	plt.close();

	with open("seafloor_extrap_" + str(my_buffer) + "_" + str(nearest[k]) + ".csv", 'w') as outfile:
		for i in range( mi ):
			for j in range( mj ):
				outfile.write("{},{},{}\n".format( x[j], y[i], seafloor[i][j][k]) )
