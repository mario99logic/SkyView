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
import scipy.stats as stats
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib import ticker

#colors palettes
import seaborn as sns

#cd C:\Frenkel\Braude\2022-2023\kosmos\vova\
#python "C:\Frenkel\Braude\2022-2023\kosmos\vova\prog.py"

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

	def matrix_rotation_AroundX_get(alphaRad):
		A=[]
		A.append([1,0,0])
		A.append([0,math.cos(alphaRad),-math.sin(alphaRad)])
		A.append([0,math.sin(alphaRad),math.cos(alphaRad)])
		return A
	def matrix_rotation_AroundY_get(alphaRad):
		A=[]
		A.append([math.cos(alphaRad),0,math.sin(alphaRad)])
		A.append([0,1,0])
		A.append([-math.sin(alphaRad),0,math.cos(alphaRad)])
		return A
	def matrix_rotation_AroundZ_get(alphaRad):
		A=[]
		A.append([math.cos(alphaRad),-math.sin(alphaRad),0])
		A.append([math.sin(alphaRad),math.cos(alphaRad),0])
		A.append([0,0,1])
		return A

	def vector3_length(self,x,y,z):
		return math.sqrt(x*x+y*y+z*z)
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
		alpha_rad=2*math.pi*t_sec/period_sec
		x=(R_km+h_km)*math.cos(longitude_rad)
		y=(R_km+h_km)*math.sin(longitude_rad)
		z=(R_km+h_km)*math.cos(latitude_rad)
		V=[[x],[y],[z]]
		A=self.matrix_rotation_AroundY_get(gamma_rad)
		AV=self.matrix_multiply(A,V)
		B=self.matrix_rotation_AroundZ_get(alpha_rad)
		BAV=self.matrix_multiply(B,AV)
		x0_km,y0_km,z0_km,alpha_rad=self.centerOfEarth_get(t_sec)
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
class clSkyObjects():
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
		def __init__(self,sName,x0_km,y0_km,z0_km,vx0_kmPerSec,vy0_kmPerSec,vz0_kmPerSec,radius_km,mass_Gton):
			self.sName=sName
			self.x0_km=x0_km
			self.y0_km=y0_km
			self.z0_km=z0_km
			self.vx0_kmPerSec=vx0_kmPerSec
			self.vy0_kmPerSec=vy0_kmPerSec
			self.vz0_kmPerSec=vz0_kmPerSec
			self.radius_km=radius_km
			self.mass_Gton=mass_Gton
			
			self.x_km=x0_km
			self.y_km=y0_km
			self.z_km=z0_km
			self.vx_kmPerSec=vx0_kmPerSec
			self.vy_kmPerSec=vy0_kmPerSec
			self.vz_kmPerSec=vz0_kmPerSec
		def XYZ_heleocentric_equatorial_get(self):
			epsilon_degree=23
			V=[[self.x_km],[self.y_km],[self.z_km]]
			MyMath=clMyMath()
			M=MyMath.matrix_rotation_AroundX_get(MyMath.angleRad_get(epsilon_degree))
			MV=MyMath.matrix_multiply(M,V)
			return MV[0][0],MV[1][0],MV[2][0]
		def XYZ_geocentric_equatorial_get(self,xOfEarth_ecliptic_km,yOfEarth_ecliptic_km,zOfEarth_ecliptic_km):#x_km,y_km,z_km=
			epsilon_degree=23
			V=[[self.x_km-xOfEarth_ecliptic_km],[self.y_km-yOfEarth_ecliptic_km],[self.z_km-zOfEarth_ecliptic_km]]
			MyMath=clMyMath()
			M=MyMath.matrix_rotation_AroundX_get(MyMath.angleRad_get(epsilon_degree))
			MV=MyMath.matrix_multiply(M,V)
			return MV[0][0],MV[1][0],MV[2][0]
		def centerOfEarth_get(self,t_sec):#x_km,y_km,z_km,alpha_rad=
			RR_km=151000000
			#phase0_degree=0
			period_sec=365.4*24*60*60
			#MyMath=clMyMath()
			alpha_rad=2*math.pi*t_sec/period_sec#+MyMath.angleRad_get(phase0_degree)
			x_km=RR_km*math.cos(alpha_rad)
			y_km=RR_km*math.sin(alpha_rad)
			z_km=0
			return x_km,y_km,z_km,alpha_rad
		def XYZ_geocentric_equatorial_by_t_sec_get(self,t_sec):#x_km,y_km,z_km=
			x_km,y_km,z_km,alpha_rad=self.centerOfEarth_get(t_sec)
			return self.XYZ_geocentric_equatorial_get(x_km,y_km,z_km)
		def R__declination_degree__rightAscension__hourAngle_degree__get(self,t_sec):#R_km, delta_degree,alpha_degree,hourAngle=
			x_km,y_km,z_km=XYZ_geocentric_equatorial_by_t_sec_get(self,t_sec)
			#x=r*cos(alpha)cos(delta)
			#y=r*sin(alpha)cos(delta)
			#z=r*sin(delta)
			MyMath=clMyMath()
			R_km=MyMath.vector3_length(x_km,y_km,z_km)
			if R<1:
				return R_km,0,0,0
			delta_rad=math.asin(z_km/R_km)
			delta_degree=angleDegree_get(delta_rad)
			cd=math.cos(delta_rad)
			if cd==0:
				return R_km, delta_degree, 0,0
			ca=float(x_km)/cd
			sa=float(y_km)/cd
			alpha_rad=math.atan2(sa,ca)
			alpha_degree=angleDegree_get(alpha_rad)
			hourAngle=float(alpha_degree*24)/360#hours
			return R_km, delta_degree, alpha_degree,hourAngle
	def Sun_get(self,t_UTC=datetime.now()):#delta_degree,alpha_degree=
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
	def testSun0(self):
		t0=datetime(2023, 1, 1, 0, 0, 0, 0)
		for d in range(365):
			t_UTC=t0+timedelta(days=d)
			delta_declination_degree,alpha_rightAscension_degree=self.Sun_get(t_UTC)#:#delta_degree,alpha_degree=
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
	def LHA_degree_get(self,GST_degree,alpha_rightAscension_degree,cameraPos_longitude_degree):
		#local hour angle
		#https://en.wikipedia.org/wiki/Hour_angle
		#return GST_degree+cameraPos_longitude_degree-alpha_rightAscension_degree
		
		#Vova: cameraPos_longitude_degree multiplied by 2 to compensate utc usage instead of local astronomic time
		return GST_degree+cameraPos_longitude_degree-alpha_rightAscension_degree
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
		GST_degree=float(tHfrom0to24*360)/24+180+partOfYearFrom0to1*360
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
	def vStar_get(self):
		vStar=[]
		
		#delta_declination_degree=1.0690905628626544, alpha_rightAscension_degree=2.4673822725948087
		#https://rwoconne.github.io/rwoclass/astr1230/table-of-brightest-stars-Cosmobrain.html
		TableOfTxt=clTableOfTxt()
		sFileName="../Objects/stars.txt"
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
	def z_zeniteDistance_degree__A_azimuth_degree_get(self,
			delta_declination_degree,
			alpha_rightAscension_degree,
			t,#our time
			bPrint=False):#z_zeniteDistance_degree,A_azimuth_degree=
		SkyObjects=clSkyObjects()
		t0=datetime(t.year, t.month, t.day, 0, 0, 0, 0)
		dt_sec=(t-t0).total_seconds()
		dt_hour=float(dt_sec)/(60*60)
		tHfrom0to24=dt_hour+self.cameraTime_dHoursReativelyToUTC
		
		t00=datetime(t.year-1, 12, 22, 0, 0, 0, 0)
		dt_sec=(t-t00).total_seconds()
		tSecYear=365.25*24*60*60
		partOfYearFrom0to1=float(dt_sec-0.25*tSecYear)/tSecYear#poka
		return SkyObjects.z_zeniteDistance_degree__A_azimuth_degree__byLongitudeLatitude_get(
			self.cameraPos_longitude_degree,
			self.cameraPos_latitude_degree,
			delta_declination_degree,
			tHfrom0to24,#utc hour of day
			partOfYearFrom0to1,
			alpha_rightAscension_degree,
			bPrint)
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
		delta_declination_degree,alpha_rightAscension_degree=SkyObjects.Sun_get(t_UTC)#:#delta_degree,alpha_degree=
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
vStar=SkyObjects.vStar_get()

#vt=[datetime(2024, 5, 2, 10, 0, 0, 0)]
vt=[]
Drewing.drewAll(vStar,vSolarSystemObject,vt=vt)
