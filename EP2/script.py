import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import sem, t
from scipy import mean

colors = ['r','b','g']
test_num = 30
confidence = 0.95

alg0_res = [] #seq
alg1_res = [] #cpu
alg2_res = [] #gpu

results = [alg0_res,alg1_res,alg2_res]

#seq
for image_heigh in [512,2048]:
	image_width = image_heigh
	threads = 4

	alg0_res = [] #seq
	for j in range(0,test_num):
		times = os.popen('./mbrot 0.27085 0.004640 0.27100 0.004810 {} {} seq 1 mb.png'.format(image_width,image_heigh)).read()
		alg0_res.append(float(times))

	alg1_res = [] #cpu
	for j in range(0,test_num):
		times = os.popen('./mbrot 0.27085 0.004640 0.27100 0.004810 {} {} cpu {} mb.png'.format(image_width,image_heigh,threads)).read()
		alg1_res.append(float(times))

	for gpu_threads in [128, 512]:
		alg2_res = [] #gpu

		results = [alg0_res,alg1_res,alg2_res]

		#gpu
		for j in range(0,test_num):
			times = os.popen('./mbrotgpu 0.27085 0.004640 0.27100 0.004810 {} {} gpu {} mb.png'.format(image_width,image_heigh,gpu_threads)).read()
			results[2].append(float(times))

		algs = ["sequencial","com opemp","na gpu"]
		#graficos
		plt.title("Comparação dos tempos dos 3 algoritmos com imagem {}x{}".format(image_width,image_heigh))
		plt.suptitle("cpu com {} threads e gpu com {} threads por bloco".format(threads, gpu_threads))
		for i in range(0,3):
			y = np.array(results[i])
			plt.hist(y, color=colors[i], histtype = 'step');

		plt.ylabel('frequência')
		plt.xlabel('tempo (em segundos)')
		plt.savefig('Graf/Comp3Alg{}x{}th{}gputh{}.png'.format(image_width, image_heigh,threads,gpu_threads))
		# plt.show()

		#estatisticas
		print("imagem {}x{}".format(image_width, image_heigh))
		print("{} threads na cpu e {} threads por bloco na gpu".format(threads, gpu_threads))
		for i in range(0,3):
			data = results[i]
			n = len(data)
			m = mean(data)
			std_err = sem(data)
			h = std_err * t.ppf((1 + confidence) / 2, n - 1)
			print("Media do alg {}: {}".format(algs[i], m))
			print("Intervalo de confiança: [{} ; {}]\n".format(m-h,m+h))
			print("")

		plt.clf()

# agora com double
for image_heigh in [512,2048]:
	image_width = image_heigh
	threads = 4

	alg0_res = [] #seq
	for j in range(0,test_num):
		times = os.popen('./mbrot 0.27085 0.004640 0.27100 0.004810 {} {} seq 1 mb.png double'.format(image_width,image_heigh)).read()
		alg0_res.append(float(times))

	alg1_res = [] #cpu
	for j in range(0,test_num):
		times = os.popen('./mbrot 0.27085 0.004640 0.27100 0.004810 {} {} cpu {} mb.png double'.format(image_width,image_heigh,threads)).read()
		alg1_res.append(float(times))

	for gpu_threads in [128,512]:
		alg2_res = [] #gpu

		results = [alg0_res,alg1_res,alg2_res]

		#gpu
		for j in range(0,test_num):
			times = os.popen('./mbrotgpu 0.27085 0.004640 0.27100 0.004810 {} {} gpu {} mb.png double'.format(image_width,image_heigh,gpu_threads)).read()
			results[2].append(float(times))

		algs = ["sequencial","com opemp","na gpu"]
		#graficos
		plt.title("Comparação dos tempos dos 3 algoritmos com imagem {}x{} e Double.".format(image_width,image_heigh))
		plt.suptitle("cpu com {} threads e gpu com {} threads por bloco".format(threads, gpu_threads))
		for i in range(0,3):
			y = np.array(results[i])
			plt.hist(y, color=colors[i], histtype = 'step');

		plt.ylabel('frequência')
		plt.xlabel('tempo (em segundos)')
		plt.savefig('Graf/Comp3AlgDouble{}x{}th{}gputh{}.png'.format(image_width, image_heigh, threads, gpu_threads))
		# plt.show()

		#estatisticas
		print("imagem {}x{}".format(image_width, image_heigh))
		print("{} threads na cpu, {} threads por bloco na gpu e DOUBLE".format(threads, gpu_threads))
		for i in range(0,3):
			data = results[i]
			n = len(data)
			m = mean(data)
			std_err = sem(data)
			h = std_err * t.ppf((1 + confidence) / 2, n - 1)
			print("Media do alg {}: {}".format(algs[i], m))
			print("Intervalo de confiança: [{} ; {}]\n".format(m-h,m+h))
		plt.clf()