import math#for calculation
import random#to simulate random values etc.
import os#to work with files
import sys #to get parameters from command line

from datetime import datetime,timedelta

import matplotlib.pyplot as plt
import matplotlib as mp
#pip install matplotlib
#pip install statistics
#import statistics
import numpy as np

from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import scipy.stats as stats
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib import ticker

#colors palettes
import seaborn as sns

class clTableOfTxt():
	def __init__(self):
		self.rows=[]
		self.shapka=self.clRow()
		self.nRows=0
		self.nCols=0
		
		self.vColText=[]
		self.vCol_iCol=-1
	class clRow():
		def __init__(self):
			self.vCellText=[]
			self.id=-1
		def readRowFromString(self,s,bFirstColNoNeedToRead):
			ss=s.split('\n')#max split = 1
			s1=ss[0]
			self.vCellText=s1.split('\t')
			if bFirstColNoNeedToRead:
				self.vCellText=self.vCellText[1:]
		def s_get(self):
			ic=0
			s=""
			for sc in self.vCellText:
				if ic>0:
					s+="\t"
				s+=sc
				ic+=1
			return s
	def printToFile(self,sFileName,bShapka):
		f=open(sFileName,'w')
		if bShapka:
			s="iRow"
			for s1 in self.shapka.vCellText:
				s+="\t"+s1
			f.write(s+"\n")
		iRow=0
		for row in self.rows:
			s=str(iRow)
			for s1 in row.vCellText:
				s+="\t"+s1
			f.write(s+"\n")
			iRow+=1
		f.close
	def readFromFile(self,sFileName,bShapka,bFirstColNoNeedToRead):
		self.rows=[]
		self.shapka=self.clRow()
		self.nRows=0
		self.nCols=0
		f=open(sFileName,'r')
		#usually newline=chr(10)="\n"=<LF> in Linux, chr(13)+chr(10)=<CR>+<LF> in windows
		n=0
		s1=""
		for s in f:
			s1=s
			n+=1
		f.close()
		f=open(sFileName,'r') if n>1 else s1.split(chr(13))
		for s in f:
			#print "832"
			if bShapka:
				self.shapka.readRowFromString(s,bFirstColNoNeedToRead)
				self.nCols=len(self.shapka.vCellText)
				#print str(self.nCols)
				bShapka=False
			else:
				row=self.clRow()
				row.readRowFromString(s,bFirstColNoNeedToRead)
				nColsInRow=len(row.vCellText)
				if nColsInRow>self.nCols:
					self.nCols=nColsInRow
				#print row.vCellText
				self.rows.append(row)
				self.nRows+=1
				#print str(self.nRows)
		if n>1:
			f.close
	def vColText_make(self,vCol_iCol):
		self.vCol_iCol=vCol_iCol
		self.vColText=[]
		#print "vCol_iCol="+str(vCol_iCol)
		for row in self.rows:
			self.vColText.append(row.vCellText[vCol_iCol])
			#print "vCol_iCol="+row.vCellText[vCol_iCol]
			pass
	def iRow_get(self,textCol,iCol):
		if iCol!=self.vCol_iCol:
			self.vColText_make(iCol)
		iRow=-1
		if textCol in self.vColText:
			iRow=self.vColText.index(textCol)
		return iRow
	def iRows_get(self,textCol,iCol):
		if iCol!=self.vCol_iCol:
			self.vColText_make(iCol)
		
		#def get_indexes(x, xs): 
		get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
		return get_indexes(textCol,self.vColText)
	def sWithoutChr10Chr13(self,s0):
		s=""
		for s1 in s0:
			if s1!=chr(10) and s1!=chr(13):
				s+=s1
		return s
	def sChangeAllTabsToSpace(self,s0):
		s=""
		for s1 in s0:
			if s1!="\t":
				s+=s1
			else:
				s+=" "
		return s
	def sNoFinishingSpaces(self,s0):
		sSS=""
		s=""
		for s1 in s0:
			if s1!=" ":
				if len(sSS)>0:
					s+=sSS
					sSS=""
				s+=s1
			else:
				sSS+=" "
		return s
	def sChangeGroupOfSpacesToSingleTab(self,s0):
		sSS=""
		s=""
		for s1 in s0:
			if s1!=" ":
				if len(sSS)>0:
					s+="\t"
					sSS=""
				s+=s1
			else:
				sSS+=" "
		if len(sSS)>0:
			s+="\t"
		return s

class clMyMath():
	def coordinatesOnScreen_CameraIn000_directedOnX_onLeftYonTopZ_byDecart_get(self,x,y,z,wX_rad=math.pi/6,wY_rad=math.pi/6,d_meters=0.03):#bOk,xOnScreen,yOnScreen=
		#wX,wY - max angles, let say Pi/6=30 grad
		if x<d_meters:
			return False,0,0
		xOnScreen_alpha=-math.atan(y/x)
		yOnScreen_alpha=math.atan(z/x)
		xOnScreen=-d*y/x
		yOnScreen=d*z/x
		b=(abs(xOnScreen_alpha)<=wX_rad)and(abs(yOnScreen_alpha)<=wY_rad)
		return b,xOnScreen,yOnScreen
	def coordinatesOnScreen_CameraIn000_directedOnlongitude0Latitude_westOnTheRightNorthOnTheTop_byAngles_get(self,longitudeRad,latitudeRad,wX_rad=math.pi/6,wY_rad=math.pi/6,d_meters=0.03):#bOk,xOnScreen,yOnScreen=
		#wX,wY - max angles, let say Pi/6=30 grad
		if abs(longitudeRad)>=math.pi/2-0.1 or abs(latitudeRad)>=math.pi/2-0.1:
			return False,0,0
		xOnScreen_alpha=-longitudeRad
		yOnScreen_alpha=latitudeRad
		xOnScreen=-d*math.tan(longitudeRad)
		yOnScreen=d*math.tan(latitudeRad)
		b=(abs(xOnScreen_alpha)<=wX_rad)and(abs(yOnScreen_alpha)<=wY_rad)
		return b,xOnScreen,yOnScreen
	def coordinatesOnScreen_CameraInX0Y0Z0_directed_alphaRadOnRight_betaRadOnTop_byDecart_get(self,x,y,z,x0,y0,z0,alphaRadOnRight,betaRadOnTop,wX_rad=math.pi/6,wY_rad=math.pi/6,d_meters=0.03):#bOk,xOnScreen,yOnScreen=
		dx=x-x0
		dy=y-y0
		dz=z-z0
		A=self.matrix_rotation_AroundZ_get(alphaRadOnRight)
		B=self.matrix_rotation_AroundY_get(betaRadOnTop)
		C=[[dx],[dy],[dz]]
		AC=self.matrix_multiply(A,C)
		BAC=self.matrix_multiply(B,AC)
		x1=BAC[0][0]
		y1=BAC[1][0]
		z1=BAC[2][0]
		return self.coordinatesOnScreen_CameraIn000_directedOnX_onLeftYonTopZ_byDecart_get(x1,y1,z1,wX_rad=wX_rad,wY_rad=wY_rad,d_meters=d_meters)
	def coordinatesOnScreen_CameraIn000_directed_alphaRadOnRight_betaRadOnTop_byAngles_get(self,longitudeRad,latitudeRad,alphaRadOnRight,betaRadOnTop,wX_rad=math.pi/6,wY_rad=math.pi/6,d_meters=0.03):#bOk,xOnScreen,yOnScreen=
		return self.coordinatesOnScreen_CameraIn000_directedOnlongitude0Latitude_westOnTheRightNorthOnTheTop_byAngles_get(longitudeRad+alphaRadOnRight,latitudeRad-betaRadOnTop,wX_rad=wX_rad,wY_rad=wY_rad,d_meters=d_meters)
	#matrices (from https://integratedmlai.com/matrixinverse/)
	def matrix_print(self, Title, M):
		print(Title)
		for row in M:
			print([round(x,3)+0 for x in row])
	def matrix_printToFile_v1(self,M,sFileName,vsRow=[],vsCol=[],s00=""):
		nRows = len(M)
		nCols = len(M[0])
		f=open(sFileName,'w')
		
		s=""
		if len(vsCol)>0:
			if len(vsRow)>0:
				s+=s00
			for i in range(nCols):
				if len(vsRow)>0 or i>0:
					s+="\t"
				if i<len(vsCol):
					s+=vsCol[i]
			f.write(s+"\n")
		
		for iRow in range(nRows):
			s=""
			if len(vsRow)>0:
				if iRow<len(vsRow):
					s+=vsRow[iRow]
			for i in range(nCols):
				if len(vsRow)>0 or i>0:
					s+="\t"
				s+=str(M[iRow][i])
			f.write(s+"\n")
		f.close()
	def matrix_pair_print(self, sAction, Title1, M1, Title2, M2):
		print(sAction)
		print(Title1, '\t'*int(len(M1)/2)+"\t"*len(M1), Title2)
		for i in range(len(M1)):
			row1 = ['{0:+7.3f}'.format(x) for x in M1[i]]
			row2 = ['{0:+7.3f}'.format(x) for x in M2[i]]
			print(row1,'\t', row2)
	def matrix_0(self, rows, cols):#A=
		A = []
		for i in range(rows):
			A.append([])
			for j in range(cols):
				A[-1].append(0.0)
		return A
	def matrix_copy(self,M):#A=
		rows = len(M)
		cols = len(M[0])
		MC = self.matrix_0(rows, cols)
		for i in range(rows):
			for j in range(rows):
				MC[i][j] = M[i][j]
		return MC
	def matrix_multiply(self,A,B):#C=
		rowsA = len(A)
		colsA = len(A[0])
		
		rowsB = len(B)
		colsB = len(B[0])
		if colsA != rowsB:
			print('Number of A columns must equal number of B rows.')
			sys.exit()
		C = self.matrix_0(rowsA, colsB)
		for i in range(rowsA):
			for j in range(colsB):
				total = 0
				for ii in range(colsA):
					total += A[i][ii] * B[ii][j]
				C[i][j] = total
		return C
	#from https: //stackoverflow.com/questions/32114054/matrix-inversion-without-numpy
	def matrix_gaussManipulations(self,A):#B=
		def eliminate(r1, r2, col, target=0):
			fac = (r2[col]-target) / r1[col]
			for i in range(len(r2)):
				r2[i] -= fac * r1[i]
		n=len(A)
		for i in range(n):
			if A[i][i] == 0:
				for j in range(i+1, n):
					if A[i][j] != 0:
						A[i], A[j] = A[j], A[i]
						break
				else:
					print("MATRIX NOT INVERTIBLE")
					return -1
			for j in range(i+1, n):
				eliminate(A[i], A[j], i)
		for i in range(n-1, -1, -1):
			for j in range(i-1, -1, -1):
				eliminate(A[i], A[j], i)
		for i in range(n):
			eliminate(A[i], A[i], i, target=1)
		return A
	def inverse(a):
	
		tmp = [[] for _ in a]
		for i,row in enumerate(a):
			assert len(row) == len(a)
			tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
		gauss(tmp)
		ret = []
		for i in range(len(tmp)):
			ret.append(tmp[i][len(tmp[i])//2:])
		return ret
	def matrix_printToFile(self,sFileName,vvDistMatrixInd,bName=True,vsCol=[],vsRow=[],sCoor="ind",viRow=[],viCol=[]):
		def i_get(i,vi):
			if len(vi)==0:
				return i
			else:
				return vi[i]
		f=open(sFileName,'w')
		nR=len(vvDistMatrixInd)
		nC=len(vvDistMatrixInd[0])
		if bName:
			s=sCoor
			if len(vsCol)>=nC:
				for i0 in range(nC):
					j=i_get(i0,viCol)
					s+="\t"+vsCol[j]
			else:
				for i0 in range(nC):
					j=i_get(i0,viCol)
					s+="\t"+str(j)
		f.write(s+"\n")
		for i0 in range(nR):
			i=i_get(i0,viRow)
			s=""
			if bName:
				if len(vsRow)>=nR:
					s=vsRow[i]
				else:
					s=str(i)
			for i1 in range(nC):
				if bName or i1>0:
					s+="\t"
				j=i_get(i1,viCol)
				s+=str(vvDistMatrixInd[i][j])
			f.write(s+"\n")
		f.close()

	def matrix_rotation_AroundX_get(self,alphaRad):
		A=[]
		A.append([1,0,0])
		A.append([0,math.cos(alphaRad),-math.sin(alphaRad)])
		A.append([0,math.sin(alphaRad),math.cos(alphaRad)])
		return A
	def matrix_rotation_AroundY_get(self,alphaRad):
		A=[]
		A.append([math.cos(alphaRad),0,math.sin(alphaRad)])
		A.append([0,1,0])
		A.append([-math.sin(alphaRad),0,math.cos(alphaRad)])
		return A
	def matrix_rotation_AroundZ_get(self,alphaRad):
		A=[]
		A.append([math.cos(alphaRad),-math.sin(alphaRad),0])
		A.append([math.sin(alphaRad),math.cos(alphaRad),0])
		A.append([0,0,1])
		return A

	def vector3_length(self,x,y,z):
		return math.sqrt(x*x+y*y+z*z)
	def vector_length(self,v):
		s=0
		for x in v:
			s+=x*x
		return math.sqrt(s)
	def angleRad_get(self,angle_degree):
		return float(angle_degree*2*math.pi)/360
	def angleDegree_get(self,angle_rad):
		return float(angle_rad*360)/(2*math.pi)
	def pointOnEarth_get(self,t_sec,longitude_degree=35,latitude_degree=32.5,h_km=0):#x_km,y_km,z_km=
		R_km=6371
		gamma_degree=23
		gamma_rad=self.angleRad_get(gamma_degree)
		longitude_rad=self.angleRad_get(longitude_degree)
		latitude_rad=self.angleRad_get(latitude_degree)
		period_sec=23*60*60+56*60+4
		t_sec=0#poka
		
		
		
		alpha_rad=2*math.pi*t_sec/period_sec
		x=(R_km+h_km)*math.cos(longitude_rad)
		y=(R_km+h_km)*math.sin(longitude_rad)
		z=(R_km+h_km)*math.cos(latitude_rad)
		V=[[x],[y],[z]]
		A=self.matrix_rotation_AroundY_get(gamma_rad)
		AV=self.matrix_multiply(A,V)
		B=self.matrix_rotation_AroundZ_get(alpha_rad)
		BAV=self.matrix_multiply(B,AV)
		x0_km,y0_km,z0_km,alpha_rad=self.centerOfEarth_get(t)
		x_km=x0_km+BAV[0][0]
		y_km=y0_km+BAV[1][0]
		z_km=z0_km+BAV[2][0]
		return x_km,y_km,z_km
	def drewObjects(self):
		pass
	def nDaysFromJ2000_get(self,t_UTC):
		#the number of days (positive or negative, including fractional days) since Greenwich noon, Terrestrial Time, on 1 January 2000
		
		#current_dateTime = datetime.now()
		# datetime(year, month, day, hour, minute, second, microsecond)
		t0=datetime(2000, 1, 1, 0, 0, 0, 0)
		dts=(t_UTC-t0).total_seconds()#no total_days() in python
		return float(dts)/(24*60*60)
	def mean(self,vv):#mean=
		n=0
		sum=0
		for a in vv:
			n+=1
			sum+=a
		if n==0:
			return 0
		return float(sum)/n
	def var(self,vv):#S^2=
		n=len(vv)
		if n<2:
			return 0
		m=self.mean(vv)
		sum=0
		for a in vv:
			sum+=(a-m)*(a-m)
		return float(sum)/(n-1)
	def STDV(self,vv):
		return(math.sqrt(self.var(vv)))
	def yByLin(self,x1,y1,x2,y2,x):#y=
		#y=ax+b, y1=a*x1+b, y2=a*x2+b => (y-y1)/(y2-y1)=(x-x1)/(x2-x1)
		if y2==y1:
			return y1
		if x2==x1:
			return 0.5*(y1+y2)
		return y1+float((y2-y1)*(x-x1))/(x2-x1)
	def angleDegreeDiff_m180_180_get(self,aDegree,bDegree):
		d=bDegree-aDegree
		if d>180:
			d-=360
		if d<=-180:
			d+=360
		return d
class clSkyObjects():
	def __init__(self):
		self.au_1000000km=149.5978707#astronomic unit (dist from the Sun to the Earth)
		#1000,000 km=10^9 m
		#1000 km=10^6 m
	class clObservation():
		def __init__(self,t,RightAscension_degree,declination_degree):
			self.t=t
			self.RightAscension_degree=RightAscension_degree
			self.declination_degree=declination_degree
	class clStar():
		#ecliptic coordinate system:
		#right ascension = longitude_ecliptic_degree
		#declination = latitude_ecliptic_degree
		def __init__(self,sName,RightAscension_degree,declination_degree):
			self.sName=sName
			self.RightAscension_degree=RightAscension_degree
			self.declination_degree=declination_degree
			self.sColor="black"
			self.size=1
	class clSolarSystemObject():
		#ecliptic coordinate system
		#see https://en.wikipedia.org/wiki/Astronomical_coordinate_systems
		def __init__(self,sName,x0_1000000km,y0_1000000km,z0_1000000km, vx0_kmPerSec,vy0_kmPerSec,vz0_kmPerSec,radius_1000km,mass_10p25kg):
			self.sName=sName
			self.x0_1000000km=x0_1000000km
			self.y0_1000000km=y0_1000000km
			self.z0_1000000km=z0_1000000km
			self.vx0_kmPerSec=vx0_kmPerSec
			self.vy0_kmPerSec=vy0_kmPerSec
			self.vz0_kmPerSec=vz0_kmPerSec
			self.radius_1000km=radius_1000km
			self.mass_10p25kg=mass_10p25kg
			
			self.x_1000000km=x0_1000000km
			self.y_1000000km=y0_1000000km
			self.z_1000000km=z0_1000000km
			self.vx_kmPerSec=vx0_kmPerSec
			self.vy_kmPerSec=vy0_kmPerSec
			self.vz_kmPerSec=vz0_kmPerSec
		def XYZ_heleocentric_equatorial_get(self):
			epsilon_degree=23
			V=[[self.x_1000000km],[self.y_1000000km],[self.z_1000000km]]
			MyMath=clMyMath()
			M=MyMath.matrix_rotation_AroundX_get(MyMath.angleRad_get(epsilon_degree))
			MV=MyMath.matrix_multiply(M,V)
			return MV[0][0],MV[1][0],MV[2][0]
		def XYZ_geocentric_equatorial_get(self,xOfEarth_ecliptic_1000000km,yOfEarth_ecliptic_1000000km,zOfEarth_ecliptic_1000000km):#x_1000000km,y_1000000km,z_1000000km=
			epsilon_degree=23
			V=[[self.x_1000000km-xOfEarth_ecliptic_1000000km],[self.y_1000000km-yOfEarth_ecliptic_1000000km],[self.z_1000000km-zOfEarth_ecliptic_1000000km]]
			MyMath=clMyMath()
			M=MyMath.matrix_rotation_AroundX_get(MyMath.angleRad_get(epsilon_degree))
			MV=MyMath.matrix_multiply(M,V)
			return MV[0][0],MV[1][0],MV[2][0]
		def centerOfEarth_get(self,t):#x_1000000km,y_1000000km,z_1000000km,alpha_rad=
			RR_1000000km=151
			#phase0_degree=0
			period_sec=365.25*24*60*60
			#MyMath=clMyMath()
			Drewing=clDrewing()
			tHfrom0to24,partOfYearFrom0to1=Drewing.tHfrom0to24__partOfYearFrom0to1_get(t)
			ppp=0.75#0.75 is found experementaly
			ppp=0.5
			t_sec=(ppp+partOfYearFrom0to1)*period_sec
			alpha_rad=2*math.pi*t_sec/period_sec#+MyMath.angleRad_get(phase0_degree)
			x_1000000km=RR_1000000km*math.cos(alpha_rad)
			y_1000000km=RR_1000000km*math.sin(alpha_rad)
			z_1000000km=0
			return x_1000000km,y_1000000km,z_1000000km,alpha_rad
		def centerOfPlanet_get(self,t,RR_1000000km=151,periodDays=365.25):#x_1000000km,y_1000000km,z_1000000km,alpha_rad=
			#phase0_degree=0
			period_sec=periodDays*24*60*60
			#MyMath=clMyMath()
			Drewing=clDrewing()
			tHfrom0to24,partOfYearFrom0to1=Drewing.tHfrom0to24__partOfYearFrom0to1_get(t)
			ppp=0.75#0.75 is found experementaly
			ppp=0.5
			t_sec=(ppp+partOfYearFrom0to1)*365.25*24*60*60
			alpha_rad=2*math.pi*t_sec/period_sec#+MyMath.angleRad_get(phase0_degree)
			print(t_sec)
			print(period_sec)
			print("t_sec/period_sec="+str(t_sec/period_sec))
			x_1000000km=RR_1000000km*math.cos(alpha_rad)
			y_1000000km=RR_1000000km*math.sin(alpha_rad)
			z_1000000km=0
			return x_1000000km,y_1000000km,z_1000000km,alpha_rad
		def XYZ_geocentric_equatorial_by_t_sec_get(self,t):#x_1000000km,y_1000000km,z_1000000km=
			x_1000000km,y_1000000km,z_1000000km,alpha_rad=self.centerOfEarth_get(t)
			return self.XYZ_geocentric_equatorial_get(x_1000000km,y_1000000km,z_1000000km)
		def R__declination_degree__rightAscension__hourAngle_degree__get(self,t):#R_1000000km, delta_degree,alpha_degree,hourAngle=
			x_1000000km,y_1000000km,z_1000000km=self.XYZ_geocentric_equatorial_by_t_sec_get(t)
			#x=r*cos(alpha)cos(delta)
			#y=r*sin(alpha)cos(delta)
			#z=r*sin(delta)
			MyMath=clMyMath()
			R_1000000km=MyMath.vector3_length(x_1000000km,y_1000000km,z_1000000km)
			if R_1000000km<0.000001:
				return R_1000000km,0,0,0
			delta_rad=math.asin(z_1000000km/R_1000000km)
			delta_degree=MyMath.angleDegree_get(delta_rad)
			cd=math.cos(delta_rad)
			if cd==0:
				return R_1000000km, delta_degree, 0,0
			ca=float(x_1000000km)/(R_1000000km*cd)
			sa=float(y_1000000km)/(R_1000000km*cd)
			alpha_rad=math.atan2(sa,ca)
			alpha_degree=MyMath.angleDegree_get(alpha_rad)
			hourAngle=float(alpha_degree*24)/360#hours
			return R_1000000km, delta_degree, alpha_degree,hourAngle
	def Sun__delta_degree__alpha_degree_get(self,t_UTC=datetime.now()):#delta_degree,alpha_degree=
		#https://en.wikipedia.org/wiki/Position_of_the_Sun
		MyMath=clMyMath()
		n=MyMath.nDaysFromJ2000_get(t_UTC)
		L=280.460+0.9856474*n
		g=357.528+0.9856003*n
		gRad=MyMath.angleRad_get(g)
		lambla=L+1.915*math.sin(gRad)+0.020*math.sin(2*gRad)
		beta=0
		LambdaRad=MyMath.angleRad_get(lambla)
		sinLambda=math.sin(LambdaRad)
		cosLambla=math.cos(LambdaRad)
		R_dist_astronomicUnits=1.00014-0.01671*math.cos(gRad)-0.00014*math.cos(2*gRad)
		epsilon=23.439-0.0000004*n #Obliquity of the ecliptic
		eRad=MyMath.angleRad_get(epsilon)
		sinE=math.sin(eRad)
		cosE=math.cos(eRad)
		#Right ascension:
		alphaRad=math.atan2(cosE*sinLambda,cosLambla)
		alpha_degree=MyMath.angleDegree_get(alphaRad)
		# declination
		deltaRad=math.asin(sinE*sinLambda)
		delta_degree=MyMath.angleDegree_get(deltaRad)
		
		#results:
		#dateime				delta					alpha
		#2023-03-21 00:00:00     0.24380412240597088     0.5624512490142688
		#2023-06-21 00:00:00     23.435517108472563      89.86966872608619
		#2023-09-22 00:00:00     0.3056052778692936      179.29496332257017
		#2023-12-21 00:00:00     -23.433877544005398     -90.71335103620048
		return delta_degree,alpha_degree
	def Sun_get(self):
		return clSkyObjects.clSolarSystemObject(
			sName="Sun",
			x0_1000000km=0,y0_1000000km=0,z0_1000000km=0,
			vx0_kmPerSec=0,vy0_kmPerSec=0,vz0_kmPerSec=0,
			radius_1000km=695.51,
			mass_10p25kg=1.98*100000)
	def testSun0(self):
		t0=datetime(2023, 1, 1, 0, 0, 0, 0)
		for d in range(365):
			t_UTC=t0+timedelta(days=d)
			delta_declination_degree,alpha_rightAscension_degree=self.Sun__delta_degree__alpha_degree_get(t_UTC)#:#delta_degree,alpha_degree=
			print(str(t_UTC)+"\t"+str(delta_declination_degree)+"\t"+str(alpha_rightAscension_degree))
	def testSun1(self):
		#dateime				delta					alpha
		#2023-03-21 00:00:00     0.24380412240597088     0.5624512490142688
		delta_declination_degree=0
		alpha_rightAscension_degree=0
		
		cameraPos_longitude_degree=0#Grinvich
		#
		cameraPos_latitude_degree=0#equator
		cameraPos_latitude_degree=90#North poluse
		cameraPos_latitude_degree=60#petrozovodsk
		#
		cameraPos_longitude_degree=35#Karmiel
		cameraPos_latitude_degree=32.5#Karmiel

		Drewing=clDrewing()
		for tH in range(24):
			print("tH="+str(tH))
			z_zeniteDistance_degree,A_azimuth_degree=self.z_zeniteDistance_degree__A_azimuth_degree__byLongitudeLatitude_get(
				cameraPos_longitude_degree,
				cameraPos_latitude_degree,
				delta_declination_degree,
				tH,
				0,
				alpha_rightAscension_degree,
				bPrint=True)
	def testSun2(self):
		Sun=self.Sun_get()
		#xHEq,yHEq,zHEq=Sun.XYZ_heleocentric_equatorial_get()
		t=datetime(2024, 3, 22, 0, 0, 0, 0)
		t=datetime(2024, 6, 22, 0, 0, 0, 0)
		t=datetime(2024, 9, 22, 0, 0, 0, 0)
		t=datetime(2024, 12, 22, 0, 0, 0, 0)
		
		#xOfEarth_ecliptic_1000000km,yOfEarth_ecliptic_1000000km,zOfEarth_ecliptic_1000000km,alpha_rad=Sun.centerOfEarth_get(t)
		#xGEq,yGEq,zGEq=Sun.XYZ_geocentric_equatorial_get(xOfEarth_ecliptic_1000000km,yOfEarth_ecliptic_1000000km,zOfEarth_ecliptic_1000000km)
		R_1000000km, delta_degree,alpha_degree,hourAngle=Sun.R__declination_degree__rightAscension__hourAngle_degree__get(t)
		print("R_1000000km="+str(R_1000000km)
			+", delta_degree="+str(delta_degree)
			+", alpha_degree="+str(alpha_degree)
			+", hourAngle="+str(hourAngle))
	def LHA_degree_get(self,GST_degree,alpha_rightAscension_degree,cameraPos_longitude_degree):
		#local hour angle
		#https://en.wikipedia.org/wiki/Hour_angle
		#return GST_degree+cameraPos_longitude_degree-alpha_rightAscension_degree
		
		#Vova: cameraPos_longitude_degree multiplied by 2 to compensate utc usage instead of local astronomic time
		return GST_degree+cameraPos_longitude_degree-alpha_rightAscension_degree
	def GST_degree_get(self,tHfrom0to24,partOfYearFrom0to1):
		return float(tHfrom0to24*360)/24+180+partOfYearFrom0to1*360
	def z_zeniteDistance_degree__A_azimuth_degree__byLongitudeLatitude_get(self,
			longitude_degree,
			phy_latitude_degree,
			delta_declination_degree,
			tHfrom0to24,#utc hour of day
			partOfYearFrom0to1,
			alpha_rightAscension_degree,
			bPrint=False):#z_zeniteDistance_degree,A_azimuth_degree=
		#https://en.wikipedia.org/wiki/Astronomical_coordinate_system
		#Equatorial <--> horizontal
		MyMath=clMyMath()
		phyRad=MyMath.angleRad_get(phy_latitude_degree)
		deltaRad=MyMath.angleRad_get(delta_declination_degree)
		
		GST_degree=self.GST_degree_get(tHfrom0to24,partOfYearFrom0to1)
		LHA_degree=self.LHA_degree_get(GST_degree,alpha_rightAscension_degree,longitude_degree)#local hour angle
		#poka
		#LHA_degree=GST_degree-alpha_rightAscension_degree
		
		#hour angle
		tRad=MyMath.angleRad_get(LHA_degree)
		
		cosZ=math.sin(phyRad)*math.sin(deltaRad)+math.cos(phyRad)*math.cos(deltaRad)*math.cos(tRad)
		
		zRad=math.acos(cosZ)
		if bPrint:
			print("tHfrom0to24="+str(tHfrom0to24)+", LHA_degree="+str(LHA_degree)+", cosZ="+str(cosZ)+", zRad="+str(zRad))
		z_zeniteDistance_degree=MyMath.angleDegree_get(zRad)
		if math.sin(zRad)==0:
			return z_zeniteDistance_degree,0
		sA=math.cos(deltaRad)*math.sin(tRad)/math.sin(zRad)
		cA=(-math.cos(phyRad)*math.sin(deltaRad)+math.sin(phyRad)*math.cos(deltaRad)*math.cos(tRad))/math.sin(zRad)
		A_azimuthRad=math.atan2(sA,cA)
		A_azimuth_degree=MyMath.angleDegree_get(A_azimuthRad)
		if bPrint:
			print("delta_declination_degree="+str(delta_declination_degree)
				+", alpha_rightAscension_degree="+str(alpha_rightAscension_degree)
				+", z_zeniteDistance_degree="+str(z_zeniteDistance_degree)
				+", A_azimuth_degree="+str(A_azimuth_degree))
		return z_zeniteDistance_degree,A_azimuth_degree
	def delta_declination_degree__alpha_rightAscension_degree_get(self, 
			z_zeniteDistance_degree,
			A_azimuth_degree,
			longitude_degree,
			phy_latitude_degree,
			tHfrom0to24,#utc hour of day
			partOfYearFrom0to1):#delta_declination_degree,alpha_rightAscension_degree=
		#https://en.wikipedia.org/wiki/Astronomical_coordinate_systems
		MyMath=clMyMath()
		aRad=MyMath.angleRad_get(90-z_zeniteDistance_degree)
		aaRad=MyMath.angleRad_get(A_azimuth_degree)
		phyRad=MyMath.angleRad_get(phy_latitude_degree)
		ca=math.cos(aRad)
		sa=math.sin(aRad)
		caa=math.cos(aaRad)
		saa=math.sin(aaRad)
		cphy=math.cos(phyRad)
		sphy=math.sin(phyRad)
		sinDelta=sphy*sa-cphy*ca*caa
		deltaRad=math.asin(sinDelta)
		delta_declination_degree=MyMath.angleDegree_get(deltaRad)
		
		alpha_rightAscension_degree=0#poka
		if ca==0:
			alpha_rightAscension_degree=0
		else:
			d=caa*sphy+math.tan(aRad)*cphy
			if d==0 and saa==0:
				alpha_rightAscension_degree=0#impossible to detect here
			else:
				hRad=math.atan2(saa,d)
				hDegree=MyMath.angleDegree_get(hRad)
				GST_degree=self.GST_degree_get(tHfrom0to24,partOfYearFrom0to1)
				#hDegree=GST_degree+cameraPos_longitude_degree-alpha_rightAscension_degree
				alpha_rightAscension_degree=GST_degree+cameraPos_longitude_degree-hDegree
		return delta_declination_degree,alpha_rightAscension_degree
	def vStar_get(self):
		vStar=[]
		
		#delta_declination_degree=1.0690905628626544, alpha_rightAscension_degree=2.4673822725948087
		#https://rwoconne.github.io/rwoclass/astr1230/table-of-brightest-stars-Cosmobrain.html
		TableOfTxt=clTableOfTxt()
		sFileName="C:\\Frenkel\\Braude\\2022-2023\\kosmos\\vova\\stars.txt"
		#readFromFile(self,sFileName,bShapka,bFirstColNoNeedToRead)
		TableOfTxt.readFromFile(sFileName,bShapka=True,bFirstColNoNeedToRead=True)
		iCol_sName=0
		iCol_RA=8
		iCol_DEC=9
		
		
		for r in TableOfTxt.rows:
			vs=r.vCellText
			sName=vs[iCol_sName]
			
			#hh mm ss.d
			#07 45 18.9
			vsRightAscension_degree=vs[iCol_RA].split(" ")
			hh=int(vsRightAscension_degree[0])
			mm=int(vsRightAscension_degree[1])
			ss=float(vsRightAscension_degree[2])
			RightAscension_degree=hh*(float(360)/24)+mm*(float(360)/(24*60))+ss*(float(360)/(24*60*60))
			
			#gg mm ss
			#+45 59 53
			vsDeclination_degree=vs[iCol_DEC].split(" ")
			gg=int(vsDeclination_degree[0])
			mm=int(vsDeclination_degree[1])
			ss=int(vsDeclination_degree[2])
			declination_degree=gg+mm*(float(360)/(24*60))+ss*(float(360)/(24*60*60))
			
			Star=clSkyObjects.clStar(sName,RightAscension_degree,declination_degree)
			vStar.append(Star)
		return vStar
	def bIsLikeStar_get(self,vObservation):
		MyMath=clMyMath()
		vd=[]
		vsra=[]
		vcra=[]
		#t,RightAscension_degree,declination_degree
		for Observation in vObservation:
			vd.append(Observation.declination_degree)
			aRad=MyMath.angleRad_get(Observation.RightAscension_degree)
			vsra.append(math.sin(aRad))
			vcra.append(math.cos(aRad))
		md=MyMath.mean(vd)
		msra=MyMath.mean(vsra)
		mcra=MyMath.mean(vcra)
		aRad=math.atan2(msra,mcra)
		ma=MyMath.angleDegree_get(aRad)
		STDVd=MyMath.STDV(vd)
		STDVsra=MyMath.STDV(vsra)
		STDVcra=MyMath.STDV(vsra)
		b=(STDVd<0.1)and(STDVsra<0.01)and(STDVcra<0.01)
		return b,md,ma
	def indexFast_get(self,Observation1,Observation2):
		dtSec=(Observation2.t-Observation1.t).total_seconds()
		#36 degree=24*60*60=240*360=>1 degree = 240 sec = 4 minutes
		if dtSec>6*60*60:
			return -1#too much time interval => unknown
		dd=Observation2.declination_degree-Observation1.declination_degree
		dar=Observation2.RightAscension_degree-Observation1.RightAscension_degree
		if dar<-180:
			dar+=360
		if dar>180:
			dar-=360
		if abs(dd)>0.3 or abs(dar)>1:
			return 2#too fast object
		if dtSec<600:
			return -1#not anought time to make accurate decision
		if abs(dd)*float(24*60*60)/tSec>0.6 or abs(dar)*float(24*60*60)/tSec>2:
			return 2#too fast object
		if abs(dd)*float(24*60*60)/tSec<0.1 and abs(dar)*float(24*60*60)/tSec<0.1:
			return 0#slow object, star?
		return 1#may be element of sun system
	def XYZ_get(self,Observation1,Observation2,xTest,yTest,zTest):#x_1000000km,y_1000000km,z_1000000km=
		#Drewing=clDrewing()
		#tHfrom0to24_1,partOfYearFrom0to1_1=self.tHfrom0to24__partOfYearFrom0to1_get(Observation1.t)
		#tHfrom0to24_2,partOfYearFrom0to1_2=self.tHfrom0to24__partOfYearFrom0to1_get(Observation2.t)
		SolarSystemObject=clSkyObjects.clSolarSystemObject(sName="",
			x0_1000000km=0,y0_1000000km=0,z0_1000000km=0, 
			vx0_kmPerSec=0,vy0_kmPerSec=0,vz0_kmPerSec=0,
			radius_1000km=1,
			mass_10p25kg=1)
		dar=Observation2.RightAscension_degree-Observation1.RightAscension_degree
		if dar<-180:
			dar+=360
		if dar>180:
			dar-=360
		dd=Observation2.declination_degree-Observation1.declination_degree
		
		iteration=0
		SolarSystemObject.mass_10p25kg=iteration
		def myFunction(z):
			SolarSystemObject.x_1000000km=z[0]
			SolarSystemObject.y_1000000km=z[1]
			SolarSystemObject.z_1000000km=z[2]
			R1,declination_degree1,rightAscension1,hourAngle_degree1=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(Observation1.t)
			R2,declination_degree2,rightAscension2,hourAngle_degree2=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(Observation2.t)
			
			daro=rightAscension2-rightAscension1
			if daro<-180:
				daro+=360
			if daro>180:
				daro-=360
			ddo=declination_degree2-declination_degree1
			
			F=np.empty((3))
			F[0]=abs(declination_degree1-Observation1.declination_degree)
			F[1]=abs(rightAscension1-Observation1.RightAscension_degree)
			F[2]=math.sqrt((daro-dar)*(daro-dar)+(ddo-dd)*(ddo-dd))
			
			iteration=SolarSystemObject.mass_10p25kg
			print(str(iteration)+" "+str(z)+" "+str(F))
			
			
			iteration+=1
			SolarSystemObject.mass_10p25kg=iteration
			return F
				
		
		#x=r*cos(alpha)cos(delta)
		#y=r*sin(alpha)cos(delta)
		#z=r*sin(delta)
		x_1000000km,y_1000000km,z_1000000km,alpha_rad=SolarSystemObject.centerOfEarth_get(Observation1.t)
		xyz_best=[x_1000000km,y_1000000km,z_1000000km]
		
		MyMath=clMyMath()
		ff_best=myFunction(xyz_best)
		f_best=MyMath.vector3_length(ff_best[0],ff_best[1],ff_best[2])
		print("f="+str(f_best))
		
		aRad=MyMath.angleRad_get(Observation1.RightAscension_degree)
		dRad=MyMath.angleRad_get(Observation1.declination_degree)
		cacd=math.cos(aRad)*math.cos(dRad)
		sacd=math.sin(aRad)*math.cos(dRad)
		sd=math.sin(dRad)
		for d_1000000km in range(10000):#(1):#
			xyz=[x_1000000km+d_1000000km*cacd,y_1000000km+d_1000000km*sacd,z_1000000km+d_1000000km*sd]
			ff=myFunction(xyz)
			f=MyMath.vector3_length(ff[0],ff[1],ff[2])
			print("f="+str(f))
			if f>f_best:
				break
			if f<f_best:
				f_best=f
				ff_best=ff
				xyz_best=xyz
		
		iteration=0
		SolarSystemObject.mass_10p25kg=iteration
		#zGuess=np.array([0,0,0])
		zGuess=np.array([xyz_best[0],xyz_best[1],xyz_best[2]])
		z=fsolve(myFunction,zGuess)
		print("The best found solution: "+str(z))
		print("for initial variant: F(x,y,z)="+str(myFunction([xTest,yTest,zTest])))
		return z[0],z[1],z[2]
	def XYZ_test(self):
		t1=datetime(2024, 5, 4, 21, 0, 0, 0)
		t2=datetime(2024, 5, 4, 21, 10, 0, 0)
		if False:
			delta_degree1,alpha_degree1=self.Sun__delta_degree__alpha_degree_get(t1)
			delta_degree2,alpha_degree2=self.Sun__delta_degree__alpha_degree_get(t2)
			o1=clSkyObjects.clObservation(t1,alpha_degree1,delta_degree1)
			o2=clSkyObjects.clObservation(t2,alpha_degree2,delta_degree2)
			x_1000000km,y_1000000km,z_1000000km=self.XYZ_get(o1,o2,0,0,0)
			return
		
		SolarSystemObject=clSkyObjects.clSolarSystemObject(sName="",
			x0_1000000km=0,y0_1000000km=0,z0_1000000km=0, 
			vx0_kmPerSec=0,vy0_kmPerSec=0,vz0_kmPerSec=0,
			radius_1000km=1,
			mass_10p25kg=1)
		if False:
			x_1000000km1,y_1000000km1,z_1000000km1,alpha_rad1=SolarSystemObject.centerOfEarth_get(t1)
			x_1000000km2,y_1000000km2,z_1000000km2,alpha_rad2=SolarSystemObject.centerOfEarth_get(t2)
		RR_1000000km=200#151#0#
		periodDays=5000000#200#365.25#
		x_1000000km1,y_1000000km1,z_1000000km1,alpha_rad1=SolarSystemObject.centerOfPlanet_get(t1,RR_1000000km=RR_1000000km,periodDays=periodDays)
		if True:
			x_1000000km2,y_1000000km2,z_1000000km2,alpha_rad2=SolarSystemObject.centerOfPlanet_get(t2,RR_1000000km=RR_1000000km,periodDays=periodDays)
		else:
			x_1000000km2=x_1000000km1
			y_1000000km2=y_1000000km1
			z_1000000km2=z_1000000km1
			alpha_rad2=alpha_rad1
		print("x,y,z,alphaRad:")
		print(str([x_1000000km1,y_1000000km1,z_1000000km1,alpha_rad1]))
		print(str([x_1000000km2,y_1000000km2,z_1000000km2,alpha_rad2]))
		SolarSystemObject.x_1000000km=x_1000000km1
		SolarSystemObject.y_1000000km=y_1000000km1
		SolarSystemObject.z_1000000km=z_1000000km1
		R1,declination_degree1,rightAscension1,hourAngle_degree1=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(t1)
		SolarSystemObject.x_1000000km=x_1000000km2
		SolarSystemObject.y_1000000km=y_1000000km2
		SolarSystemObject.z_1000000km=z_1000000km2
		R2,declination_degree2,rightAscension2,hourAngle_degree2=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(t2)
		print("R,d,a,hDegree:")
		print(str([R1,declination_degree1,rightAscension1,hourAngle_degree1]))
		print(str([R2,declination_degree2,rightAscension2,hourAngle_degree2]))
		o1=clSkyObjects.clObservation(t1,rightAscension1,declination_degree1)
		o2=clSkyObjects.clObservation(t2,rightAscension2,declination_degree2)
		x_1000000km,y_1000000km,z_1000000km=self.XYZ_get(o1,o2,x_1000000km1,y_1000000km1,z_1000000km1)
	def XYZXYZ_get(self,
			Observation1,Observation2,Observation3,
			xTest,yTest,zTest,VxTest,VyTest,VzTest):#x_1000000km,y_1000000km,z_1000000km,vx_kmPerSec,vy_kmPerSec,vz_kmPerSec=
		SolarSystemObject=clSkyObjects.clSolarSystemObject(sName="",
			x0_1000000km=0,y0_1000000km=0,z0_1000000km=0, 
			vx0_kmPerSec=0,vy0_kmPerSec=0,vz0_kmPerSec=0,
			radius_1000km=1,
			mass_10p25kg=1)
		
		dtSec12=(Observation2.t-Observation1.t).total_seconds()
		dtSec13=(Observation3.t-Observation1.t).total_seconds()
		MyMath=clMyMath()
		bPrint=True
		
		iteration=0
		SolarSystemObject.mass_10p25kg=iteration
		def myFunction(z):
			SolarSystemObject.x_1000000km=z[0]
			SolarSystemObject.y_1000000km=z[1]
			SolarSystemObject.z_1000000km=z[2]
			R1,declination_degree1,rightAscension1,hourAngle_degree1=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(Observation1.t)
			SolarSystemObject.x_1000000km=z[3]
			SolarSystemObject.y_1000000km=z[4]
			SolarSystemObject.z_1000000km=z[5]
			R3,declination_degree3,rightAscension3,hourAngle_degree3=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(Observation3.t)
			#yByLin(self,x1,y1,x2,y2,x)
			SolarSystemObject.x_1000000km=MyMath.yByLin(0,z[0],dtSec13,z[3],dtSec12) 
			SolarSystemObject.y_1000000km=MyMath.yByLin(0,z[1],dtSec13,z[4],dtSec12) 
			SolarSystemObject.z_1000000km=MyMath.yByLin(0,z[2],dtSec13,z[5],dtSec12) 
			R2,declination_degree2,rightAscension2,hourAngle_degree2=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(Observation2.t)
			
			F=np.empty((6))
			F[0]=abs(declination_degree1-Observation1.declination_degree)
			F[1]=abs(MyMath.angleDegreeDiff_m180_180_get(rightAscension1,Observation1.RightAscension_degree))
			F[2]=abs(declination_degree2-Observation2.declination_degree)
			F[3]=abs(MyMath.angleDegreeDiff_m180_180_get(rightAscension2,Observation2.RightAscension_degree))
			F[4]=abs(declination_degree3-Observation3.declination_degree)
			F[5]=abs(MyMath.angleDegreeDiff_m180_180_get(rightAscension3,Observation3.RightAscension_degree))
			
			iteration=SolarSystemObject.mass_10p25kg
			if bPrint:
				print(str(iteration)+" "+str(z)+" "+str(F))
			
			
			iteration+=1
			SolarSystemObject.mass_10p25kg=iteration
			return F
				
		
		#x=r*cos(alpha)cos(delta)
		#y=r*sin(alpha)cos(delta)
		#z=r*sin(delta)
		x_1000000km1,y_1000000km1,z_1000000km1,alpha_rad1=SolarSystemObject.centerOfEarth_get(Observation1.t)
		x_1000000km3,y_1000000km3,z_1000000km3,alpha_rad3=SolarSystemObject.centerOfEarth_get(Observation3.t)
		xyzxyz_best=[x_1000000km1,y_1000000km1,z_1000000km1,x_1000000km3,y_1000000km3,z_1000000km3]
		
		
		ff_best=myFunction(xyzxyz_best)
		f_best=MyMath.vector_length(ff_best)
		print("f="+str(f_best))
		
		aRad1=MyMath.angleRad_get(Observation1.RightAscension_degree)
		dRad1=MyMath.angleRad_get(Observation1.declination_degree)
		cacd1=math.cos(aRad1)*math.cos(dRad1)
		sacd1=math.sin(aRad1)*math.cos(dRad1)
		sd1=math.sin(dRad1)
		
		aRad3=MyMath.angleRad_get(Observation3.RightAscension_degree)
		dRad3=MyMath.angleRad_get(Observation3.declination_degree)
		cacd3=math.cos(aRad3)*math.cos(dRad3)
		sacd3=math.sin(aRad3)*math.cos(dRad3)
		sd3=math.sin(dRad3)
		
		d_1000000km1_best=0
		d_1000000km3_best=0
		
		def ggg(d_1000000km1,d_1000000km3,f_best,ff_best,xyzxyz_best,d_1000000km1_best,d_1000000km3_best,bPrint=False):
			bOk=True
			xyzxyz=[x_1000000km1+d_1000000km1*cacd1,
				y_1000000km1+d_1000000km1*sacd1,
				z_1000000km1+d_1000000km1*sd1,
				x_1000000km3+d_1000000km3*cacd3,
				y_1000000km3+d_1000000km3*sacd3,
				z_1000000km3+d_1000000km3*sd3]
			ff=myFunction(xyzxyz)
			f=MyMath.vector_length(ff)
			if bPrint:
				print("f="+str(f))
			if f>f_best:
				bOk=False
			if f<f_best:
				f_best=f
				ff_best=ff
				xyzxyz_best=xyzxyz
				d_1000000km1_best=d_1000000km1
				d_1000000km3_best=d_1000000km3
			return bOk,f_best,ff_best,xyzxyz_best,d_1000000km1_best,d_1000000km3_best
		
		if False:
			for d_1000000km1 in range(0,10000,100):#(1):#
				for d_1000000km3 in range(0,10000,100):#(1):#
					bOk=True
					bOk,f_best,ff_best,xyzxyz_best,d_1000000km1_best,d_1000000km3_best=ggg(d_1000000km1,d_1000000km3,f_best,ff_best,xyzxyz_best,d_1000000km1_best,d_1000000km3_best)
					if not bOk:
						break
			d10=max(0,d_1000000km1-100)
			d11=d_1000000km1+100
			d30=max(0,d_1000000km3-100)
			d31=d_1000000km3+100
			for d_1000000km1 in range(d10,d11):#(1):#
				for d_1000000km3 in range(d30,d31):#(1):#
					bOk=True
					bOk,f_best,ff_best,xyzxyz_best,d_1000000km1_best,d_1000000km3_best=ggg(d_1000000km1,d_1000000km3,f_best,ff_best,xyzxyz_best,d_1000000km1_best,d_1000000km3_best)
					if not bOk:
						break
		
		bPrint=False
		for d_1000000km1 in range(10000):#(1):#
			d_1000000km3 =d_1000000km1
			bOk=True
			bOk,f_best,ff_best,xyzxyz_best,d_1000000km1_best,d_1000000km3_best=ggg(d_1000000km1,d_1000000km3,f_best,ff_best,xyzxyz_best,d_1000000km1_best,d_1000000km3_best)
			if not bOk:
				break
		bPrint=True
		
		iteration=0
		SolarSystemObject.mass_10p25kg=iteration
		#zGuess=np.array([0,0,0])
		zGuess=np.array([xyzxyz_best[0],xyzxyz_best[1],xyzxyz_best[2],xyzxyz_best[3],xyzxyz_best[4],xyzxyz_best[5]])
		z=fsolve(myFunction,zGuess)
		print("The best found solution: "+str(z))
		print()
		print()
		print("for initial variant: F(x,y,z,x,y,z)="+str(myFunction([xTest,yTest,zTest,VxTest,VyTest,VzTest])))
		return z[0],z[1],z[2],z[3],z[4],z[5]
	def XYZXYZ_test(self):
		t1=datetime(2024, 1, 4, 21, 0, 0, 0)
		t2=datetime(2024, 4, 4, 22, 10, 0, 0)
		t3=datetime(2024, 7, 4, 23, 20, 0, 0)
		if False:
			delta_degree1,alpha_degree1=self.Sun__delta_degree__alpha_degree_get(t1)
			delta_degree2,alpha_degree2=self.Sun__delta_degree__alpha_degree_get(t2)
			delta_degree3,alpha_degree3=self.Sun__delta_degree__alpha_degree_get(t3)
			o1=clSkyObjects.clObservation(t1,alpha_degree1,delta_degree1)
			o2=clSkyObjects.clObservation(t2,alpha_degree2,delta_degree2)
			o3=clSkyObjects.clObservation(t3,alpha_degree3,delta_degree3)
			x_1000000km1,y_1000000km1,z_1000000km1,x_1000000km3,y_1000000km3,z_1000000km3=self.XYZXYZ_get(o1,o2,o3,0,0,0,0,0,0)
			return
		
		SolarSystemObject=clSkyObjects.clSolarSystemObject(sName="",
			x0_1000000km=0,y0_1000000km=0,z0_1000000km=0, 
			vx0_kmPerSec=0,vy0_kmPerSec=0,vz0_kmPerSec=0,
			radius_1000km=1,
			mass_10p25kg=1)
		if False:
			x_1000000km1,y_1000000km1,z_1000000km1,alpha_rad1=SolarSystemObject.centerOfEarth_get(t1)
			x_1000000km2,y_1000000km2,z_1000000km2,alpha_rad2=SolarSystemObject.centerOfEarth_get(t2)
			x_1000000km3,y_1000000km3,z_1000000km3,alpha_rad3=SolarSystemObject.centerOfEarth_get(t3)
		RR_1000000km=200#151#0#
		periodDays=5000#0#200#365.25#
		x_1000000km1,y_1000000km1,z_1000000km1,alpha_rad1=SolarSystemObject.centerOfPlanet_get(t1,RR_1000000km=RR_1000000km,periodDays=periodDays)
		if True:
			x_1000000km2,y_1000000km2,z_1000000km2,alpha_rad2=SolarSystemObject.centerOfPlanet_get(t2,RR_1000000km=RR_1000000km,periodDays=periodDays)
			x_1000000km3,y_1000000km3,z_1000000km3,alpha_rad3=SolarSystemObject.centerOfPlanet_get(t3,RR_1000000km=RR_1000000km,periodDays=periodDays)
		else:
			x_1000000km2=x_1000000km1
			y_1000000km2=y_1000000km1
			z_1000000km2=z_1000000km1
			alpha_rad2=alpha_rad1
		print("x,y,z,alphaRad:")
		print(str([x_1000000km1,y_1000000km1,z_1000000km1,alpha_rad1]))
		print(str([x_1000000km2,y_1000000km2,z_1000000km2,alpha_rad2]))
		print(str([x_1000000km3,y_1000000km3,z_1000000km3,alpha_rad3]))
		SolarSystemObject.x_1000000km=x_1000000km1
		SolarSystemObject.y_1000000km=y_1000000km1
		SolarSystemObject.z_1000000km=z_1000000km1
		R1,declination_degree1,rightAscension1,hourAngle_degree1=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(t1)
		SolarSystemObject.x_1000000km=x_1000000km2
		SolarSystemObject.y_1000000km=y_1000000km2
		SolarSystemObject.z_1000000km=z_1000000km2
		R2,declination_degree2,rightAscension2,hourAngle_degree2=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(t2)
		SolarSystemObject.x_1000000km=x_1000000km3
		SolarSystemObject.y_1000000km=y_1000000km3
		SolarSystemObject.z_1000000km=z_1000000km3
		R3,declination_degree3,rightAscension3,hourAngle_degree3=SolarSystemObject.R__declination_degree__rightAscension__hourAngle_degree__get(t3)

		print("R,d,a,hDegree:")
		print(str([R1,declination_degree1,rightAscension1,hourAngle_degree1]))
		print(str([R2,declination_degree2,rightAscension2,hourAngle_degree2]))
		print(str([R3,declination_degree3,rightAscension3,hourAngle_degree3]))
		o1=clSkyObjects.clObservation(t1,rightAscension1,declination_degree1)
		o2=clSkyObjects.clObservation(t2,rightAscension2,declination_degree2)
		o3=clSkyObjects.clObservation(t3,rightAscension3,declination_degree3)
		x_1000000km1,y_1000000km1,z_1000000km1,x_1000000km3,y_1000000km3,z_1000000km3=self.XYZXYZ_get(o1,o2,o3,x_1000000km1,y_1000000km1,z_1000000km1,x_1000000km3,y_1000000km3,z_1000000km3)
	
class clDrewing():
	def __init__(self):
		#self.plt=plt
		self.cameraTime_dHoursReativelyToUTC=-3#Israel summer time
		self.cameraPos_longitude_degree=35#Karmiel
		self.cameraPos_latitude_degree=32.5#Karmiel
		self.cameraDirection_AzimuthSouth_degree=-38#0#view to South
		self.cameraDirection_h_degree=38#19#some up
		self.cameraView_maxHorizontalAngle_degree=40#30#70#
		self.cameraView_maxVerticalAngle_degree=40#20#
		self.cameraView_distance_cm=1#3
	def z_zeniteDistance_degree_get(self,h_elevation_degree):
		return 90-h_elevation_degree
	def h_elevation_degree_get(self,z_zeniteDistance_degree):
		return 90-z_zeniteDistance_degree
	def tHfrom0to24__partOfYearFrom0to1_get(self,t):#tHfrom0to24,partOfYearFrom0to1=
		t0=datetime(t.year, t.month, t.day, 0, 0, 0, 0)
		dt_sec=(t-t0).total_seconds()
		dt_hour=float(dt_sec)/(60*60)
		tHfrom0to24=dt_hour+self.cameraTime_dHoursReativelyToUTC
		
		t00=datetime(t.year-1, 12, 22, 0, 0, 0, 0)
		dt_sec=(t-t00).total_seconds()
		tSecYear=365.25*24*60*60
		partOfYearFrom0to1=float(dt_sec-0.25*tSecYear)/tSecYear#poka
		return tHfrom0to24,partOfYearFrom0to1
	def z_zeniteDistance_degree__A_azimuth_degree_get(self,
			delta_declination_degree,
			alpha_rightAscension_degree,
			t,#our time
			bPrint=False):#z_zeniteDistance_degree,A_azimuth_degree=
		SkyObjects=clSkyObjects()
		tHfrom0to24,partOfYearFrom0to1=self.tHfrom0to24__partOfYearFrom0to1_get(t)
		return SkyObjects.z_zeniteDistance_degree__A_azimuth_degree__byLongitudeLatitude_get(
			self.cameraPos_longitude_degree,
			self.cameraPos_latitude_degree,
			delta_declination_degree,
			tHfrom0to24,#utc hour of day
			partOfYearFrom0to1,
			alpha_rightAscension_degree,
			bPrint)
	def delta_declination_degree__alpha_rightAscension_degree_get(self, 
				z_zeniteDistance_degree,
				A_azimuth_degree,
				t):#delta_declination_degree,alpha_rightAscension_degree=
		SkyObjects=clSkyObjects()
		tHfrom0to24,partOfYearFrom0to1=self.tHfrom0to24__partOfYearFrom0to1_get(t)
		return SkyObjects.delta_declination_degree__alpha_rightAscension_degree_get(self, 
			z_zeniteDistance_degree,
			A_azimuth_degree,
			self.cameraPos_longitude_degree,
			self.cameraPos_latitude_degree,
			tHfrom0to24,#utc hour of day
			partOfYearFrom0to1)
	def drewObject(self,z_zeniteDistance_degree,A_azimuth_degree,sColor="black",size=1):
		h_local_degree=self.h_elevation_degree_get(z_zeniteDistance_degree)#-self.cameraDirection_h_degree
		A_local_degree=A_azimuth_degree#-self.cameraDirection_AzimuthSouth_degree
		#print("h_local_degree="+str(h_local_degree))
		#print("A_local_degree="+str(A_local_degree))
		if (abs(h_local_degree-self.cameraDirection_h_degree)<self.cameraView_maxVerticalAngle_degree and abs(A_local_degree-self.cameraDirection_AzimuthSouth_degree)<self.cameraView_maxHorizontalAngle_degree):
			x=float(self.cameraView_distance_cm*A_local_degree)#/self.cameraView_maxHorizontalAngle_degree
			y=float(self.cameraView_distance_cm*h_local_degree)#/self.cameraView_maxVerticalAngle_degree
			plt.scatter(x,y,s=size,c=sColor)
	def drewStar(self,Star,t):
		delta_declination_degree=Star.declination_degree
		alpha_rightAscension_degree=Star.RightAscension_degree
		z_zeniteDistance_degree,A_azimuth_degree=self.z_zeniteDistance_degree__A_azimuth_degree_get(
				delta_declination_degree,
				alpha_rightAscension_degree,
				t,
				bPrint=False)
				
		self.drewObject(z_zeniteDistance_degree,A_azimuth_degree,sColor=Star.sColor,size=Star.size)
	def drewSun(self,t,sColor="yellow",size=50,bPrint=False):
		t_UTC=t+timedelta(hours=self.cameraTime_dHoursReativelyToUTC)
		if bPrint:
			print("t="+str(t))
		delta_declination_degree,alpha_rightAscension_degree=SkyObjects.Sun__delta_degree__alpha_degree_get(t_UTC)#:#delta_degree,alpha_degree=
		z_zeniteDistance_degree,A_azimuth_degree=self.z_zeniteDistance_degree__A_azimuth_degree_get(
				delta_declination_degree,
				alpha_rightAscension_degree,
				t,
				bPrint=bPrint)
		self.drewObject(z_zeniteDistance_degree,A_azimuth_degree,sColor=sColor,size=size)
	def drewSolarSystemObject(self,SolarSystemObject):
		pass
	def drewAll(self,vStar,vSolarSystemObject,vt=[]):
		xMin=self.cameraView_distance_cm*(self.cameraDirection_AzimuthSouth_degree-self.cameraView_maxHorizontalAngle_degree)
		xMax=self.cameraView_distance_cm*(self.cameraDirection_AzimuthSouth_degree+self.cameraView_maxHorizontalAngle_degree)
		yMin=self.cameraView_distance_cm*(self.cameraDirection_h_degree-self.cameraView_maxVerticalAngle_degree)
		yMax=self.cameraView_distance_cm*(self.cameraDirection_h_degree+self.cameraView_maxVerticalAngle_degree)
		y0=self.cameraView_distance_cm*(self.cameraDirection_h_degree)
		x0=self.cameraView_distance_cm*(self.cameraDirection_AzimuthSouth_degree)
		sColor="black"
		plt.plot([xMin,xMax],[yMin,yMin],c=sColor)
		plt.plot([xMin,xMax],[yMax,yMax],c=sColor)
		plt.plot([xMin,xMin],[yMin,yMax],c=sColor)
		plt.plot([xMin,xMin],[yMin,yMax],c=sColor)
		#plt.plot([xMin,xMax],[y0,y0],c="red")
		#plt.plot([x0,x0],[yMin,yMax],c="blue")
		plt.plot([xMin,xMax],[0,0],c="red")
		plt.plot([0,0],[yMin,yMax],c="blue")
		
		if False:
			for i in range(100):
				p=1-float((i-50)*(i-50))/(40*40)
				z_zeniteDistance_degree=self.z_zeniteDistance_degree_get(70*p)
				A_azimuth_degree=-50+i
				print("i="+str(i)+", p="+str(p)+", z_zeniteDistance_degree="+str(z_zeniteDistance_degree)+", A_azimuth_degree="+str(A_azimuth_degree))
				self.drewObject(z_zeniteDistance_degree,A_azimuth_degree,sColor="black",size=i)
		
		SkyObjects=clSkyObjects()
		if len(vt)==0:
			dtHours=0.1
			tH=0
			while tH<24:
				delta_declination_degree=16#23#epsilon_degree=23
				alpha_rightAscension_degree=40#float(360*tH)/24
				#t_UTC==datetime.now()
				#t_UTC=datetime(2024, 6, 22, 0, 0, 0, 0)
				t=datetime(2024, 5, 2, 0, 0, 0, 0)
				#t=datetime(2024, 6, 22, 0, 0, 0, 0)
				#t=datetime(2024, 12, 22, 0, 0, 0, 0)
				t=datetime(2024, 3, 22, 0, 0, 0, 0)
				t+=timedelta(hours=tH)
				vt.append(t)
				tH+=dtHours
		for t in vt:
			self.drewSun(t,sColor="yellow",size=50,bPrint=False)
			for Star in vStar:
				self.drewStar(Star,t)
		plt.show()
		'''
		x_data=np.array(vx)
		y_data=np.array(vn)
		plt.plot(x_data,y_data, color="black")
		plt.legend()
		plt.show()
		'''
class clCoorTransform():
	def __init__(self):
		pass
Drewing=clDrewing()
vStar=[]
vSolarSystemObject=[]
if False:
	MyMath=clMyMath()
	t_UTC=datetime.now()
	n=MyMath.nDaysFromJ2000_get(t_UTC)
	print("n_now="+str(n))
SkyObjects=clSkyObjects()
#SkyObjects.testSun0()
#SkyObjects.testSun1()
#SkyObjects.testSun2()
#SkyObjects.XYZ_test()
SkyObjects.XYZXYZ_test()

vStar=SkyObjects.vStar_get()

#vt=[datetime(2024, 5, 2, 10, 0, 0, 0)]
vt=[]
#Drewing.drewAll(vStar,vSolarSystemObject,vt=vt)
