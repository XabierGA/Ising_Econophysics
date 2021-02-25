from scipy.stats import norm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(24)

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
			plot = plt.imshow(img[i] , cmap="viridis")
			plt.axis("off")
			plt.title("Epoch -> " +str(i))
			if i == 1000 or i == 3000 or i == 5000 or i ==7000:
				plt.savefig("epoch_" + str(i) + ".png")
			writer.grab_frame()
			plot.remove()
	plt.close("all")


def runSimulation(lattice , epochs):

	for x in range(epochs):
		print("Epoch " , x , "----> " , epochs)
		lattice.updateSpin()
	folder = "/home/xabierga/Ising_Econophysics/TMP_IMG"
	#generate_video(lattice.history , folder)
	return

def plot_results(lattice):

	fig , (ax1,ax2,ax3) = plt.subplots(3,1 , figsize=(8,12))
	plt.subplots_adjust(hspace = 0.5)
	ax1.set_title("2D Ising Model simulation , 32x32 lattice") 
	ax1.plot(range(int(lattice.total_epochs*lattice.equilibration) , lattice.total_epochs ) , lattice.magnetizations , color="black")
	ax1.set_ylabel("$M_{(t)}$")
	ax1.set_xlabel("$\Delta t$")
	ax1.legend()
	ax3.hist(np.diff(lattice.magnetizations)  , bins=45 , density=True, color="dodgerblue" , label="Empirical Distribution")
	mu , std = norm.fit(np.diff(lattice.magnetizations))
	xmin  , xmax = np.min(np.diff(lattice.magnetizations)) , np.max(np.diff(lattice.magnetizations))
	x = np.linspace(xmin , xmax , 100)
	p = norm.pdf(x , mu , std)
	ax3.plot(x , p ,'k' , linewidth=2 , label = "Gaussian Fit")
	ax3.legend(loc=2)
	ax3.set_xlabel("$R_{(t)}$")
	axins3 = zoomed_inset_axes(ax3, zoom = 5, loc=1)
	axins3.hist(np.diff(lattice.magnetizations) , bins=45 , density = True , color="dodgerblue" , label="Fat Tails")
	axins3.plot(x,p , 'k')
	axins3.legend(loc=2)
	x1, x2, y1, y2 = 0.019,0.025,0,7.5
	axins3.set_xlim(x1, x2)
	axins3.set_ylim(y1, y2)
	mark_inset(ax3, axins3, loc1=4, loc2=3, fc="none", ec="0.5")

	ax2.set_xlabel("$\Delta t$")
	ax2.set_ylabel("$R_{(t)}$")
	ax2.plot(range(int(lattice.total_epochs*Ising.equilibration) , lattice.total_epochs - 1) , np.diff(lattice.magnetizations) , color="black" )
	ax2.legend()
	plt.savefig("sim_result.pdf")
	plt.show(True)


def get_acf(lattice):
	log_returns = np.diff(lattice.magnetizations)
	plot_acf(log_returns , lags=30)
	plt.title("Simulation Log returns autocorrelation")
	plt.xlabel("Lag")
	plt.ylabel("ACF")
	plt.savefig("simulation_acf.pdf")
	plt.show()


steps = 100
ep = 10000
alpha = 15
T = 1
size = 32
Ising = SquareLattice(T , "random" , size , alpha , steps , ep , 0.2)


runSimulation(Ising , ep)


#plot_results(Ising)
get_acf(Ising)
