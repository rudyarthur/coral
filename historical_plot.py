import matplotlib.pyplot as plt

import numpy as np
import csv

import sys
from math import log
import ast

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import box

my_buffer = 25
#nn = sys.argv[1]

polygons = []
with open('/local/home/rudyarthur/coral/coral_polygons.dat', 'r') as infile:
	for line in infile:
		p = ast.literal_eval( line );
		poly = Polygon( p ).buffer(my_buffer);
		polygons.append( poly );
		
		
		
def in_poly(x, y):

	pt = Point(x, y);
	for r,p in enumerate(polygons):
		if p.contains( pt ):	
			return r;
			
	#print( "Coral not in rect" );
	#sys.exit(1);
	return -1;
	
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
	

		

        
	
growth = [ 
[ [-4000,-3500], 1.4, 0, 'seagreen' ],
[ [-3500,-3000], 5.5,  5.2, 'peru'],
[ [-3000,-2500], 6.4 , 7.8, 'g' ],
[ [-2500,-2000], 6.9 , 9.4, 'b' ],
[ [-2000,-1500], 5.1 , 4.0, 'y' ],
[ [-1500,-1000], 2.5 , 1.3, 'c' ],
[ [-1000,-500], 3.8 , 3.8, 'm' ],
[ [-500,0], 3.4 , 5.1, 'k'],
[ [0,1500], 1.4 , 1.1, 'r']
]

B=100
T = 1501;
skip=10

point_map = {}

lower_limits = []
upper_limits = []
seafloors = [1, 4, 10, 20, 50, 100, 250]

for nn in seafloors:
	for r in [0.0]:
		for err in ["Gaussian"]:

			result_file = open("/data/Coral/" + err + "_historical_coral" + "_" + str(my_buffer) + '_' + str(nn) + ".dat", 'w');	
			result_file.write("Error Patch Central Min Max\n")		
					
			coral = [ ]
			central = [ ];
			for t in range(0,T,skip):
				name = "/data/Coral/Centralheight-" + str(t) + "r" + str(r) + "_" + str(my_buffer) + '_' +  str(nn) + "_data.csv"
				print( name )
				with open(name, 'r') as csvfile:
					spamreader = csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
					csum = [ 0 for p in polygons ]
					for row in spamreader:
						coord = (float(row[0]), float(row[1]))
						if coord in point_map:
							blob = point_map[coord]
						else:
							blob = in_poly( float(row[0]), float(row[1]) )
							point_map[coord] = blob;
						if blob > -1:
							csum[ blob ] += float(row[3])
					central.append(csum);
							
				data = [];
				if err == "Central": B = 0;
					
				for i in range(B):
					if err == "Central":
						name = "/data/Coral/" + err + "height-" + str(t) + "r" + str(r) + "_" + str(my_buffer) + '_' +  str(nn) + "_data.csv"
					else:
						name = "/data/Coral/" + err + str(i) + "height-" + str(t) + "r" + str(r) + "_" + str(my_buffer) + '_' +  str(nn) + "_data.csv"
					print( name )
					with open(name, 'r') as csvfile:
						spamreader = csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
						csum = [ 0 for p in polygons ]
						for row in spamreader:
							coord = (float(row[0]), float(row[1]))
							if coord in point_map:
								blob = point_map[coord]
							else:
								blob = in_poly( float(row[0]), float(row[1]) )
								point_map[coord] = blob;
							if blob > -1:
								csum[ blob ] += float(row[3])
						data.append( csum )
				#data.sort();
				coral.append( data );	
				#print( t, len(coral) , len(coral[-1]), central[-1], ":", coral[-1] )
			#sys.exit(1);
				
			if err == "Central":
				lb = 0;
				ub = 0;
			else:
				conf = 0.95;
				lb = int((1-conf)*B)
				ub = int(conf*B)
					
			coral = np.array(coral)
			cols = [ "red", "brown", "grey", "green", "black", "yellow", "blue", "cyan", "magenta"]

			lerrs = []
			uerrs = []
			for rect,p in enumerate(polygons):
				
				time = [];
				av = []
				lerr = [];
				uerr = [];
				original = float( central[0][rect] )

				min_date = None;
				max_date = None;
				av_date = None;
				
				for i,t in enumerate(range(0,T,skip)):
					
					time.append( t )
					av.append( central[i][rect]/original )
					if av[-1] == 0 and av_date is None: av_date = t;
					
					if err != "Central":
						tmp = sorted( coral[i,:,rect] )
						lerr.append( tmp[ lb ]/original )
						uerr.append( tmp[ ub ]/original )

					
						if lerr[-1] == 0 and min_date is None: min_date = t;
						if uerr[-1] == 0 and max_date is None: max_date = t;
					else:
						#placeholder
						lerr.append( av[-1] - 0.01 )
						uerr.append( av[-1] + 0.01 )
						
				if err != "Central":
					result_file.write(err + " {} {} {} {} {}\n".format(rect, nn, av_date, min_date, max_date) )		
				else:
					result_file.write(err + " {} {} {}\n".format(rect, nn, av_date) )		

				time = np.array(time)
				lerr = np.array(lerr)
				uerr = np.array(uerr)
				lerrs.append(lerr)
				uerrs.append(uerr);
				
				#plt.plot(time, av, color = cols[rect], label="Patch " + str(rect))
				#plt.plot(time, lerr, "k-")
				#plt.plot(time, uerr, "k-")
				plt.fill_between(time, lerr, uerr, facecolor=cols[rect],  alpha=0.75, edgecolor=None )
				plt.plot([], [], color=cols[rect], linewidth=10, label="Patch " + str(rect))
		
		
			plt.legend(loc='upper right')
			plt.xlabel("Years ago")
			plt.ylabel("Proportion of present day coral")
			plt.ylim([0,1]);
			plt.savefig(err + "_historical_coral" + "_" + str(my_buffer) + '_' + str(nn) + ".eps")
			#plt.show()
			plt.close()

			result_file.close()
			lower_limits.append(lerrs)
			upper_limits.append(uerrs)

for j,p in enumerate(polygons):
	lerr = []
	uerr = []
	time = []
	for i,t in enumerate(range(0,T,skip)):
		time.append(t)
		lvs = [ lower_limits[k][j][i] for k,nn in enumerate(seafloors) ]
		uvs = [ upper_limits[k][j][i] for k,nn in enumerate(seafloors) ]
		lerr.append( min(lvs) )
		uerr.append( max(uvs) )
	time = np.array(time)
	lerr = np.array(lerr)
	uerr = np.array(uerr)
				
	plt.fill_between(time, lerr, uerr, facecolor=cols[j],  alpha=0.75, edgecolor=None )
	plt.plot([], [], color=cols[j], linewidth=10, label="Patch " + str(j))

plt.legend(loc='upper right')
plt.xlabel("Years ago")
plt.ylabel("Proportion of present day coral")
plt.ylim([0,1]);
plt.savefig(err + "_historical_coral" + "_" + str(my_buffer) + ".eps")
#plt.show()
plt.close()
		
