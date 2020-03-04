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


from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import box

import ast

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
	
def habitat(height, coral, growth):
	
	sizex, sizey = height.shape;
	habitat = np.zeros( 1+len(growth) );
	count = 0;
	count2 = 0;

	for i in range(sizex):
		for j in range(sizey):	
			#is coral
			if coral[i][j] > cl:
				count += 1.0;
				got = False
				for k,g in enumerate(growth):
					if ht[i][j] >= g[0][0] and ht[i][j] < g[0][1]:
						habitat[ 1+k ] += 1.0;
						count2 += 1;
						got = True;
						break;
				if not got:
					habitat[0] += 1;
					#print( "missed", coral[i][j], ht[i][j] )
				
	#print( count, sum(habitat) )
	return habitat, count;


##Grid of where the flipping coral is
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

"""		
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
"""

x = [];
y = [];
z = [];
#mk = [];
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
			zcoord = 0;
		else:
			zcoord = float(row[2])*1000;
		
		#if (row[0],row[1]) in coral:
		#	iscoral = 1.0;
		#else:
		#	iscoral = 0.0;
			
		if len(line) == 0:
			y.append( ycoord );
			
		if len(line) > 0 and xcoord < xlast:
			ht.append( line );
			#mk.append( cline );
			line = [ zcoord ];
			#cline = [ iscoral ];
			#print( "ydiff", ycoord - ylast );
			gotx = True;
			y.append( ycoord );
			
		else:
			line.append( zcoord );
			#cline.append( iscoral );
			if not gotx: x.append( xcoord );
			#print( "xdiff", xcoord - xlast );
			
		xlast = xcoord
		ylast = ycoord
		
ht.append( line )
#mk.append( cline )

##Convert to np arrays
ht = np.array( ht );
#mk = np.array( mk );
x = np.array(x)
y = np.array(y)

mk = np.empty( ht.shape );
mj = len(x)
mi = len(y)
for i in range( mi ):
	for j in range( mj ):
		got = False;
		pt = Point( x[j], y[i] );
		for r,p in enumerate(polygons):
			if p.contains( pt ):
				mk[i][j] = 1
				got = True;
				break;
		if not got:
			mk[i][j] = 0;
				
#CS = plt.contourf(x,y,mk, 60, cmap=plt.cm.ocean_r)
#plt.colorbar()  
#plt.show()
#sys.exit(1);

#################################################################
## Get the sea floor
#################################################################
##nearest neighbours
stag = sys.argv[5]
save_data = sys.argv[6];

	
seafloor = [];
ty = []
with open('/local/home/rudyarthur/coral/v6/seafloor_extrap_' + str(my_buffer) + '_' + stag + ".csv", 'r') as csvfile:
	spamreader = csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
	line = [];
	xlast = np.inf;
	ylast = np.inf;
	
	gotx = False;
	goty = False;
	
	for row in spamreader:
		
		xcoord = float(row[0]);
		ycoord = float(row[1]);

		if len(line) == 0:
			ty.append( ycoord );
			
		if len(line) > 0 and xcoord < xlast:
			seafloor.append( line );
			line = [ float(row[2]) ];
			ty.append( ycoord );			
		else:
			line.append( float(row[2]) );
			
		xlast = xcoord
		ylast = ycoord
seafloor.append( line )
		
seafloor = np.array( seafloor );

##Have a look
#CS = plt.contourf(x,y,seafloor, 60, cmap=plt.cm.ocean_r)
#plt.colorbar()  
#plt.show()
#sys.exit(1);


growth = [ 
[ [-4000,-3500], 1.4, 0, 'k' ],
[ [-3500,-3000], 5.5,  5.2, 'r'],
[ [-3000,-2500], 6.4 , 7.8, 'g' ],
[ [-2500,-2000], 6.9 , 9.4, 'b' ],
[ [-2000,-1500], 5.1 , 4.0, 'y' ],
[ [-1500,-1000], 2.5 , 1.3, 'c' ],
[ [-1000,-500], 3.8 , 3.8, 'm' ],
[ [-500,0], 3.4 , 5.1, 'seagreen'],
[ [0,1500], 1.4 , 1.1, 'peru']
]

##average grid spacing (x_i - x_i-1) should be same for all i anyway.
step = 0;
for i in range(1,len(x)):
	step += x[i] - x[i-1];
step /= (len(x) - 1);

lateral_growth = 0.25;
lg_step = step/lateral_growth;
lg_add = lateral_growth/step;

sizex, sizey = mk.shape


dist_size = 1000;
sea_rise_py = float(sys.argv[3]);

seed = int(sys.argv[1]);	
np.random.seed(seed=(seed+123456789))

gauss = int(sys.argv[2]);
if gauss == 0:
	err = "Central";
elif gauss == 1:
	##Gaussian
	err = "Gaussian" + str(seed);
else:
	##BadYear
	err = "BadYear" + str(seed);
	rands = [];
	for g in growth:
		r = norm.rvs(loc=g[1], scale=g[2], size=dist_size)
		r.sort();
		rands.append(r);

reverse = False;
if sys.argv[4] == "reverse":
	reverse = True;
		
if reverse:
	filename=err + "r" + str(sea_rise_py)  + "_" + str(my_buffer) + '_' + stag + "_backward"; 	
	tag = "-"
	tdir = -1
else:
	filename=err + "r" + str(sea_rise_py)  + "_" + str(my_buffer) + '_' + stag + "_forward"; 
	tag = "+"
	tdir = 1;

	
numc = None;
num = 0
cl = 0.999;
if reverse:
	for i in range(sizex):
		for j in range(sizey):	
			if mk[i][j] > cl: 
				if seafloor[i][j] > ht[i][j]:
					ht[i][j] = seafloor[i][j]; #can only go down to the seafloor
					mk[i][j] = 0; #land not coral
				#ht[i][j] = max(seafloor[i][j], ht[i][j]) #can only go down to the seafloor
	
#CS = plt.contourf(x,y,ht, 60, cmap=plt.cm.ocean_r)
#plt.colorbar()  
#plt.show()
#sys.exit(1);

if save_data == "save":
	
	#CS = plt.contourf(x,y,ht, 60, cmap=plt.cm.ocean_r, vmax=1500, vmin=-7500)
	#plt.colorbar() 			
	#plt.title(tag + str(0))
	#plt.savefig("/data/Coral/" + err + "height" + tag + str(0) + "r" + str(sea_rise_py) + "_" + str(my_buffer) + '_' + stag + ".png")
	#plt.close();
			
	with open("/data/Coral/" + err + "height" + tag + str(0) + "r" + str(sea_rise_py) + "_" + str(my_buffer) + '_' + stag + "_data.csv", 'w') as ofile:
		for j in range(sizey):			
			for i in range(sizex):
				interval = -1			
				#if mk[i][j] > cl:
				#	for k,g in enumerate(growth):
				#		if ht[i][j] >= g[0][0] and ht[i][j] <= g[0][1]:
				#			interval = len(growth) - k;				
				#			break;
				ofile.write("{}, {}, {}, {}, {}\n".format( x[j], y[i], ht[i][j], mk[i][j], interval) )

#dfile = open("/data/Coral/" + filename + ".dat", 'w')
#hab, tot = habitat(ht, mk, growth);
#hab_str = str(0) + " " + str( tot ); 
#for h in hab: hab_str += " " + str(h);
#dfile.write( hab_str + "\n" )

T = 1501;
skip=10

for t in range(T):
	#BadYear
	if gauss > 1:
		year_idx = np.random.random_integers(0, high=(dist_size-1))

	num = 0;
	##vertical growth
	#maxh = -4000;
	for i in range(sizex):
		for j in range(sizey):	
			if mk[i][j] >= cl:
				if (not reverse) or (ht[i][j] > seafloor[i][j] and ht[i][j] > -4000):
					
					#if ht[i][j] > maxh: maxh = ht[i][j];
					
					for k,g in enumerate(growth):
						if ht[i][j] >= g[0][0] and ht[i][j] <= g[0][1]:

							if gauss == 0:
								ht[i][j] += g[1]*tdir;	
							elif gauss == 1:
								ht[i][j] += norm.rvs(loc=g[1], scale=g[2])*tdir #Gaussian1
							else:
								ht[i][j] += rands[k][year_idx]*tdir; #BadYear
						
							num += 1;						
							break;
							
				#else:
				#	mk[i][j] = 0;
			if reverse:
				if ht[i][j] < seafloor[i][j]:
					ht[i][j] = seafloor[i][j];
					mk[i][j] = 0;
				if ht[i][j] < -4000:
					mk[i][j] = 0;
						
	if numc is None: numc = float( num );
	
	##horizontal growth	
	if not reverse:
		new_coral = np.zeros( mk.shape );
		for i in range(sizex):
			for j in range(sizey):	
				#not coral
				if mk[i][j] < cl:
					# neighbour is coral
					if i+1 < sizex and mk[i+1][j] >= cl: new_coral[i][j] += lg_add;
					if i-1 >= 0    and mk[i-1][j] >= cl: new_coral[i][j] += lg_add;
					if j+1 < sizey and mk[i][j+1] >= cl: new_coral[i][j] += lg_add;
					if j-1 >= 0    and mk[i][j-1] >= cl: new_coral[i][j] += lg_add;

		for i in range(sizex):
			for j in range(sizey):	
				mk[i][j] += new_coral[i][j];
				
		for i in range(sizex):
			for j in range(sizey):	
				ht[i][j] -= sea_rise_py*tdir;

						
	##Make habitat plot
	#hab, tot = habitat(ht, mk, growth);
	#print(t, tot)
	#hab_str = str(t+1) + " " + str( tot ); 
	#for h in hab: hab_str += " " + str(h);
	#dfile.write( hab_str + "\n" );
	
	if (t+1) % skip == 0:

		if save_data == "save":

			#CS = plt.contourf(x,y,ht, 60, cmap=plt.cm.ocean_r, vmax=1500, vmin=-7500)
			#plt.colorbar() 			
			#plt.title(tag + str(t+1))
			#plt.savefig("/data/Coral/" + err + "height" + tag + str(t+1) + "r" + str(sea_rise_py) + "_" + str(my_buffer) + '_' + stag + ".png")
			#plt.close();
		
			with open("/data/Coral/" + err + "height" + tag + str(t+1) + "r" + str(sea_rise_py) + "_" + str(my_buffer) + '_' + stag + "_data.csv", 'w') as ofile:
				for j in range(sizey):			
					for i in range(sizex):
						interval = -1			
						#if mk[i][j] > cl:
						#	for k,g in enumerate(growth):
						#		if ht[i][j] >= g[0][0] and ht[i][j] <= g[0][1]:
						#			interval = len(growth) - k;				
						#			break;
						ofile.write("{}, {}, {}, {}, {}\n".format( x[j], y[i], ht[i][j], mk[i][j], interval) )

#dfile.close()
