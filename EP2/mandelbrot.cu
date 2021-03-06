
#include <iostream>
#include <sys/time.h>
#include <string>
#include <complex>
#include "png++/png.hpp"

int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y) {
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

using namespace std;

template<class real> // para trocar entre float e double
void mandelbrot_seq(char *argv[]){
	real c0r = stod(argv[1]);
	real c0i = stod(argv[2]);
	real c1r = stod(argv[3]);
	real c1i = stod(argv[4]);

	int W = stoi(argv[5]);
	int H = stoi(argv[6]);

	int threads = stoi(argv[8]);

	string saida = argv[9];

	real real_step = (c1r - c0r)/W;
	real imag_step = (c1i - c0i)/H;

	png::image< png::rgb_pixel > imagem(W, H);

	for (png::uint_32 y = 0; y < imagem.get_height(); ++y){
		for (png::uint_32 x = 0; x < imagem.get_width(); ++x){
				real point_r = c0r+x*real_step;
		        real point_i = c0i+y*imag_step;

				// printf("%f %f\n", point_r, point_i);
		    	const int M = 1000;

				// valor Zj que falhou
				// -1 se não tiver falhado
				int j = -1;

				//Valor da iteração passada
				real old_r = 0;
				real old_i = 0;
				real aux = 0;

				//Calcula o mandebrot
				for(int i = 1; i <= M; i++){

					//Calculo da nova iteração na mão
					aux = (old_r * old_r) - (old_i * old_i) + point_r;
					old_i = (2 * old_r * old_i) + point_i;
					old_r = aux;

					//abs(complex) = sqrt(a*a + b*b)
					//Passei a raiz do abs para outro lado
					if( ((old_r * old_r + old_i * old_i) > 4 )){
						j = i;
						break;
					}
				}

			if (j == -1){
				imagem.set_pixel(x, y, png::rgb_pixel(0, 0, 0));
			}
			else{
				png::uint_32 r = (M-j*255)/M;
				png::uint_32 g = (M-j*239)/M + 16;
				png::uint_32 b = (M-j*191)/M + 64;
				imagem.set_pixel(x, y, png::rgb_pixel(r, g, b));
			}
		}
	}

	imagem.write(saida);
}


template<class real> // para trocar entre float e double
void mandelbrot_omp(char *argv[]){
	real c0r = stod(argv[1]);
	real c0i = stod(argv[2]);
	real c1r = stod(argv[3]);
	real c1i = stod(argv[4]);

	int W = stoi(argv[5]);
	int H = stoi(argv[6]);

	int threads = stoi(argv[8]);

	string saida = argv[9];

	real real_step = (c1r - c0r)/W;
	real imag_step = (c1i - c0i)/H;

	png::image< png::rgb_pixel > imagem(W, H);
	png::uint_32 y;
	png::uint_32 x;

	#pragma omp parallel for collapse(2) num_threads(threads)
		for (y = 0; y < imagem.get_height(); ++y){
			for (x = 0; x < imagem.get_width(); ++x){
				real point_r = c0r+x*real_step;
		        real point_i = c0i+y*imag_step;

				// printf("%f %f\n", point_r, point_i);
		    	const int M = 1000;

				// valor Zj que falhou
				// -1 se não tiver falhado
				int j = -1;

				//Valor da iteração passada
				real old_r = 0;
				real old_i = 0;
				real aux = 0;

				//Calcula o mandebrot
				for(int i = 1; i <= M; i++){

					//Calculo da nova iteração na mão
					aux = (old_r * old_r) - (old_i * old_i) + point_r;
					old_i = (2 * old_r * old_i) + point_i;
					old_r = aux;

					//abs(complex) = sqrt(a*a + b*b)
					//Passei a raiz do abs para outro lado
					if( ((old_r * old_r + old_i * old_i) > 4 )){
						j = i;
						break;
					}
				}

				if (j == -1){
					imagem.set_pixel(x, y, png::rgb_pixel(0, 0, 0));
				}
				else{
					png::uint_32 r = (M-j*255)/M;
					png::uint_32 g = (M-j*239)/M + 16;
					png::uint_32 b = (M-j*191)/M + 64;
					imagem.set_pixel(x, y, png::rgb_pixel(r, g, b));
				}
			}
		}

	imagem.write(saida);
}

//Função que furtei do add.cu
void cudaAssert(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("Erro %d!\n", err);
        abort();
    }
}


__global__ void gpu_calculation(REAL c0r, REAL c0i, REAL REAL_step, REAL imag_step, REAL *results, unsigned n, int W, int H){

	// index = m*x + y
	const long unsigned globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

	// printf("%d %d\n", blockIdx.x, threadIdx.x);

	if (globalIndex < n) {
        //calcular os complexos na mão
		int x = globalIndex/W;
		int y = globalIndex%H;
		// printf("%d %d    %d\n", x, y, n);
        REAL point_r = c0r+x*REAL_step;
        REAL point_i = c0i+y*imag_step;

		// printf("%f %f\n", point_r, point_i);
    	const int M = 1000;

		// valor Zj que falhou
		// -1 se não tiver falhado
		int j = -1;

		//Valor da iteração passada
		REAL old_r = 0;
		REAL old_i = 0;
		REAL aux = 0;

		//Calcula o mandebrot
		for(int i = 1; i <= M; i++){

			//Calculo da nova iteração na mão
			aux = (old_r * old_r) - (old_i * old_i) + point_r;
			old_i = (2 * old_r * old_i) + point_i;
			old_r = aux;

			//abs(complex) = sqrt(a*a + b*b)
			//Passei a raiz do abs para outro lado
			if( ((old_r * old_r + old_i * old_i) > 4 )){
				j = i;
				break;
			}
		}
		// printf("%d\n", j);
		// printf("%d\n", j);

		results[globalIndex] = j;
		// printf("%d\n", j);
	}
	// else printf("oh boy\n");

}

// para trocar entre float e double
void mandelbrot_gpu(char *argv[]){
	REAL c0r = stof(argv[1]);
	REAL c0i = stof(argv[2]);
	REAL c1r = stof(argv[3]);
	REAL c1i = stof(argv[4]);

	int W = stoi(argv[5]);
	int H = stoi(argv[6]);

	int threads = stoi(argv[8]);

	string saida = argv[9];

	REAL REAL_step = (c1r - c0r)/W;
	REAL imag_step = (c1i - c0i)/H;

	//printf("step gpu %f %f %f %f\n", c1r, c1i, c0r, c0i);
	//printf("step gpu %f %f %f %f\n", REAL_step, imag_step,(c1r - c0r), (c1i - c0i));

	png::image< png::rgb_pixel > imagem(W, H);
	// png::uint_32 y;
	// png::uint_32 x;

	//Cuda Stuff
	const int THREADS_PER_BLOCK = 128;
	const int NUM_BLOCKS = (W*H + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
	//Guarda os resultados calculados na GPU
	REAL *results = new REAL[W*H];

	//Coisas da memoria do cuda
	//Descomentar caso de erro <<<<<<<<<

	REAL *cuda_results;

	//Aloca tudo
	//cudaAssert(cudaMalloc(&cu_c0r, sizeof(REAL*)));
	//cudaAssert(cudaMalloc(&cu_c0i, sizeof(REAL*)));
	//cudaAssert(cudaMalloc(&cu_REAL_step, sizeof(REAL*)));
	//cudaAssert(cudaMalloc(&cu_imag_step, sizeof(REAL*)));

	cudaAssert(cudaMalloc(&cuda_results, W*H*sizeof(REAL)));

	//Copia tudo
	// cudaAssert(cudaMemcpy((void**)c0r_p, cu_c0r, sizeof(*c0r_p), cudaMemcpyHostToDevice));
	// cudaAssert(cudaMemcpy((void**)c0i_p, cu_c0i, sizeof(*c0i_p), cudaMemcpyHostToDevice));
	// cudaAssert(cudaMemcpy((void**)c1r_p, cu_REAL_step, sizeof(*c1r_p), cudaMemcpyHostToDevice));
	// cudaAssert(cudaMemcpy((void**)c1i_p, cu_imag_step, sizeof(*c1i_p), cudaMemcpyHostToDevice));


	//Dois problemas
	//1: não sei se templates funcionam
	//2: não sei se mandar diretamente c0r/etc funciona
	//printf("%f %f\n", REAL_step, imag_step);
	gpu_calculation<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(c0r, c0i, REAL_step, imag_step, cuda_results, W*H, W, H);

	//Pega os resultados do Cuda e desaloca
	//printf("vo copia\n");
	cudaAssert(cudaMemcpy(results, cuda_results, W*H*sizeof(REAL), cudaMemcpyDeviceToHost));

	//cudaFree(cu_c0r);
	//cudaFree(cu_c0i);
	//cudaFree(cu_REAL_step);
	//cudaFree(cu_imag_step);

	cudaFree(cuda_results);

	const int M = 1000;
	int j; //para ficar parecido aos outros

	for(int x = 0; x < H; x++){
		for(int y = 0; y < W; y++){
			//Acho que o X e o Y é algo assim
			// int x = p/N;
			// int y = p%N;

			j = results[W*x + y];
			// if (j!=23)
				// printf("%d:%d - %d\n",x,y, j);

			if (j == -1)
				imagem.set_pixel(x, y, png::rgb_pixel(0, 0, 0));
			else {
				png::uint_32 r = (M-j*255)/M;
				png::uint_32 g = (M-j*239)/M + 16;
				png::uint_32 b = (M-j*191)/M + 64;
				imagem.set_pixel(x, y, png::rgb_pixel(r, g, b));
			}
		}

	}
	//printf("copiei\n");

	imagem.write(saida);

	//Sempre bom desalocar
	delete [] results;
}

int main(int argc, char *argv[]){
	//processar os args
	//mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>

	if(argc < 10){
		cout << "Incorrect Number of Args" << endl;
		cout << "Usage:" << endl;
		cout << "mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>" << endl;
		return 0;
	}

	struct timeval t1, t2, t3;

	gettimeofday(&t1, NULL);

	string cgpu(argv[7]);
	if(cgpu == "cpu")
		mandelbrot_omp<float>(argv);
	else if(cgpu == "gpu")
		mandelbrot_gpu(argv);
	else if(cgpu == "seq")
		mandelbrot_seq<float>(argv);
	else
		cout << "Errrooou";

	gettimeofday(&t2, NULL);

    timeval_subtract(&t3, &t2, &t1);

    printf("%lu.%06lu\n", t3.tv_sec, t3.tv_usec);

	return 0;

}
