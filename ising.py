from scipy.stats import norm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import subprocess

sns.set_style("darkgrid")


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
		return (mat[(i+1)%self.size,j] + mat[(i-1 + self.size)%self.size,j] + mat[i, (j+1)%self.size] + mat[i , (j-1 + self.size)%self.size]) - self.alpha*self.lattice[i,j]*np.abs(mag) , mag

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
			self.history.append(np.copy(self.lattice))
		self.epoch += 1
#		self.history.append(self.lattice)



def generate_video(img , folder):

	ffmpeg = animation.writers["ffmpeg"]
	writer = ffmpeg(fps=100)
	fig = plt.figure()
	with writer.saving(fig , folder + "/test_video.mp4" , 100):
		for i in range(len(img)):
			print("Image -> " + str(i) + " / " + str(len(img)))
			plot = plt.imshow(img[i])
			plt.axis("off")
			plt.title("Epoch -> " +str(i))
			writer.grab_frame()
			plot.remove()
	plt.close("all")


def runSimulation(lattice , epochs):

	for x in range(epochs):
		print("Epoch " , x , "----> " , epochs)
		lattice.updateSpin()
		#print(lattice.lattice)
	folder = "/home/xabierga/Ising_Econophysics/TMP_IMG"
	#generate_video(lattice.history , folder)
	return




steps = 100
ep = 10000
alpha = 20
T = 1
size = 32
Ising = SquareLattice(T , "random" , size , alpha , steps , ep , 0.2)


runSimulation(Ising , ep)

fig , (ax1,ax2,ax3) = plt.subplots(1,3 , figsize=(20,10))
#plt.plot(range(int(Ising.total_epochs*Ising.equilibration) , Ising.total_epochs  -1) , np.diff(np.log(np.abs(Ising.magnetizations) + 0.0000001)) ,"k")
ax1.plot(range(int(Ising.total_epochs*Ising.equilibration) , Ising.total_epochs ) , Ising.magnetizations)
ax3 = sns.distplot(np.diff(Ising.magnetizations) , kde = False , fit = norm)
ax2.plot(range(int(Ising.total_epochs*Ising.equilibration) , Ising.total_epochs - 1) , np.diff(Ising.magnetizations))
plt.show(True)
