import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import sem, t
from scipy import mean

colors = ['r','b','g']
image_width = 1024
image_heigh = 1024
confidence = 0.95

alg0_res = [] #seq
alg1_res = [] #cpu
alg2_res = [] #gpu

results = [alg0_res,alg1_res,alg2_res]

os.popen('make')

#seq
for j in range(0,20):
	times = os.popen('./mbrot 0.27085 0.004640 0.27100 0.004810 {} {} seq 1 mb.png'.format(image_width,image_heigh)).read()
	results[0].append(float(times))

#cpu
for j in range(0,20):
	times = os.popen('./mbrot 0.27085 0.004640 0.27100 0.004810 {} {} cpu 8 mb.png'.format(image_width,image_heigh)).read()
	results[1].append(float(times))

#gpu
#não sei como rodar na gpu

plt.title("Comparação inicial dos tempos dos 3 algoritmos.")
for i in range(0,2): #até 3 quando tiver a gpu
	y = np.array(results[i])
	plt.hist(y, color=colors[i], histtype = 'step');

plt.savefig('Graf/Comp3Alg.png')
plt.show()

algs = ["sequencial","com opemp","na gpu"]	
for i in range(0,2): #até 3 quando tiver a gpu
	data = results[i]
	n = len(data)
	m = mean(data)
	std_err = sem(data)
	h = std_err * t.ppf((1 + confidence) / 2, n - 1)
	print("Media do alg {}: {}".format(algs[i], m))
	print("Intervalo de confiança: [{} ; {}]\n".format(m-h,m+h))