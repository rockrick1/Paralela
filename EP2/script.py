import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import sem, t
from scipy import mean

colors = ['r','b','g']
image_width = 512
image_heigh = 512
test_num = 20
confidence = 0.95

alg0_res = [] #seq
alg1_res = [] #cpu
alg2_res = [] #gpu

results = [alg0_res,alg1_res,alg2_res]


#seq
print('testando sequencial')
for j in range(0,test_num):
	times = os.popen('./mbrot 0.27085 0.004640 0.27100 0.004810 {} {} seq 1 mb.png'.format(image_width,image_heigh)).read()
	results[0].append(float(times))

#cpu
print('testando paralelo')
for j in range(0,test_num):
	times = os.popen('./mbrot 0.27085 0.004640 0.27100 0.004810 {} {} cpu 8 mb.png'.format(image_width,image_heigh)).read()
	results[1].append(float(times))

#gpu
print('testando gpu')
for j in range(0,test_num):
	times = os.popen('./mbrotgpu 0.27085 0.004640 0.27100 0.004810 {} {} gpu 8 mb.png'.format(image_width,image_heigh)).read()
	results[2].append(float(times))

algs = ["sequencial","com opemp","na gpu"]
#graficos
plt.title("Comparação dos tempos dos 3 algoritmos com imagem {}x{}.".format(image_width,image_heigh))
for i in range(0,3):
	y = np.array(results[i])
	plt.hist(y, color=colors[i], histtype = 'step');

plt.savefig('Graf/Comp3Alg.png')
plt.show()

#estatisticas
for i in range(0,3):
	data = results[i]
	n = len(data)
	m = mean(data)
	std_err = sem(data)
	h = std_err * t.ppf((1 + confidence) / 2, n - 1)
	print("Media do alg {}: {}".format(algs[i], m))
	print("Intervalo de confiança: [{} ; {}]\n".format(m-h,m+h))