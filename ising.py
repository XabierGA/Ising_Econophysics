import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import subprocess 
#from IPython.display import clear_output
#import 


class SquareLattice:


	def __init__(self, T , initial_conditions , N , alpha , steps):

		self.size = N
 		self.T = T
		self.initial = initial_conditions
		self.alpha = alpha
		self.steps = steps
		self.epoch = 0
		self.history = []
		self.magnetizations = []
		self.lattice = self._createLattice()

	def _createLattice(self):

		if self.initial == "random":
			return np.random.choice([1 , -1],(self.size,self.size))

		elif self.initial == "positive":
			return np.full((self.size, self.size) , 1)

		elif self.initial == "negative":
			return np.full((self.size, self.size) , -1)


	def calculateMag(self):

		return np.sum(np.sum(self.lattice))/(self.size**2)

	def calculateH(self , i , j):

		mag = self.calculateMag()
		self.magnetizations.append(mag)
		mat = self.lattice
		return mat[i,j]*(mat[(i+1)%self.size,j] + mat[(i-1 + self.size)%self.size,j] + mat[i, (j+1)%self.size] + mat[i , (j-1 + self.size)%self.size]) - self.alpha*self.lattice[i,j]*np.abs(mag)

	def updateSpin(self, i, j):

		beta = 1/self.T
		for x in range(self.steps):

				h = self.calculateH(i,j)
				p = 1/(1+np.exp(-2*beta*h))
				z = np.random.rand()
				self.lattice[i,j] = np.sign(p -z)
#		print(plt.imshow(self.lattice).make_image(renderer="None"))

		self.history.append(self.lattice)



def generate_video(img , folder):


	for i in range(len(img)):
		plt.imshow(img[i])
		plt.axis("off")
		plt.title("Epoch -> " +str(i))
		plt.savefig(folder + "/file%02d.png" % i)

	os.chdir(folder)
	subprocess.call(['ffmpeg', '-framerate', '100', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p','video_name.mp4'])
	for file_name in glob.glob("*.png"):
		os.remove(file_name)


def runSimulation(lattice , epochs):

	for x in range(epochs):
		print("Epoch " , x , "----> " , epochs)
		i , j= np.random.randint(0 , lattice.size , 2)
		lattice.updateSpin(i,j)
		#print(lattice.lattice)
	#folder = "/home/xabierga/Ising_Econophysics/TMP_IMG"
	#generate_video(lattice.history , folder)
	return




steps = 50
Ising = SquareLattice(1.5 , "random" , 32 , 4 , 50)
ep = 600
print(Ising.lattice)
runSimulation(Ising , ep)
print(np.diff(Ising.magnetizations))
plt.plot(range(ep*steps - 1) , np.diff(np.log(np.abs(Ising.magnetizations) + 0.0000001)) ,"k")
plt.show(True)
