import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import subprocess 
#from IPython.display import clear_output
#import 


class SquareLattice:


	def __init__(self, T , initial_conditions , N , alpha , steps , total_epochs , equilibration):

		self.size = N
		self.T = T
		self.initial = initial_conditions
		self.alpha = alpha
		self.steps = steps
		self.equilibration = equilibration
		self.epoch = 0
		self.total_epochs = total_epochs
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
		mat = self.lattice
		return mat[i,j]*(mat[(i+1)%self.size,j] + mat[(i-1 + self.size)%self.size,j] + mat[i, (j+1)%self.size] + mat[i , (j-1 + self.size)%self.size]) - self.alpha*self.lattice[i,j]*mag , mag

	def updateSpin(self):

		beta = 1/self.T
		for x in range(self.steps):
				i,j = np.random.randint(0 , self.size , 2)
				h , mag = self.calculateH(i,j)
				p = 1/(1+np.exp(-2*beta*h))
				z = np.random.rand()
				self.lattice[i,j] = np.sign(p -z)

		if self.epoch >= int(self.total_epochs*self.equilibration):
			self.magnetizations.append(mag)
			self.history.append(self.lattice)
		self.epoch += 1
#		self.history.append(self.lattice)



def generate_video(img , folder):


	for i in range(len(img)):
		print("Image -> " + str(i) + " / " + str(len(img)))
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
		lattice.updateSpin()
		#print(lattice.lattice)
	folder = "/home/xabierga/Ising_Econophysics/TMP_IMG"
	generate_video(lattice.history , folder)
	return




steps = 100
ep = 10000
alpha = 20
T = 1
size = 32
Ising = SquareLattice(T , "random" , size , alpha , steps , ep , 0.95)


runSimulation(Ising , ep)


plt.plot(range(int(Ising.total_epochs*Ising.equilibration) , Ising.total_epochs  -1) , np.diff(np.log(np.abs(Ising.magnetizations) + 0.0000001)) ,"k")
plt.show(True)
