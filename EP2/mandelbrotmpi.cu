#include <mpi.h>
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

///////////////////////////////////////SEQ//////////////////////////////////////
template<class real> // para trocar entre float e double
void mandelbrot_seq(char *argv[]){
	real c0r = stof(argv[1]);
	real c0i = stof(argv[2]);
	real c1r = stof(argv[3]);
	real c1i = stof(argv[4]);

	int W = stoi(argv[5]);
	int H = stoi(argv[6]);

	real real_step = (c1r - c0r)/W;
	real imag_step = (c1i - c0i)/H;

	//printf("step gpu %f %f %f %f\n", c1r, c1i, c0r, c0i);
	//printf("step gpu %f %f %f %f\n", real_step, imag_step,(c1r - c0r), (c1i - c0i));
	int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = (W*H)/world_size;
    int inicial;
    int r = (W*H)%world_size;
    if(rank < r){
    	inicial =  rank*size + rank;
    	size++;
    }
    else inicial = rank*size + r;
	int *results = new int[size];
    for(int k = 0; k < size; k++){
    	int x = (k + inicial)/W;
		int y = (k + inicial)%H;
		// printf("%d %d    %d\n", x, y, n);
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
		// printf("%d\n", j);
		// printf("%d\n", j);

		results[k] = j;
		// printf("%d\n", j);
	}
    if(rank == 0){
		png::image< png::rgb_pixel > imagem(W, H);
		string saida = argv[9];
		const int M = 1000;
		int j; //para ficar parecido aos outros
		int m_size = size;
		for(int i = 0; i < world_size; i++){

			int m_inicial = i*(W*H)/world_size;
			if(i < r) m_inicial += i;
			else{
				m_inicial += r
				m_size = (W*H)/world_size;
			}
			if(i != 0) MPI_Recv(results, m_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			for(int k = 0; k < m_size; k++) {
			//Acho que o X e o Y é algo assim
			// int x = p/N;
			// int y = p%N;

				j = results[k];
                int x, y;
                x = (m_inicial + k)/H;
                y = (m_inicial + k)%W;
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
		imagem.write(saida);
	}
	else{
		MPI_Send(results, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	//printf("copiei\n");


	//Sempre bom desalocar
	delete [] results;
}


///////////////////////////////////////OMP//////////////////////////////////////
template<class real> // para trocar entre float e double
void mandelbrot_omp(char *argv[]){
	real c0r = stof(argv[1]);
	real c0i = stof(argv[2]);
	real c1r = stof(argv[3]);
	real c1i = stof(argv[4]);

	int W = stoi(argv[5]);
	int H = stoi(argv[6]);

	int threads = stoi(argv[8]);

	real real_step = (c1r - c0r)/W;
	real imag_step = (c1i - c0i)/H;

	// printf("step gpu %f %f %f %f\n", c1r, c1i, c0r, c0i);
	//printf("step gpu %f %f %f %f\n", real_step, imag_step,(c1r - c0r), (c1i - c0i));
	int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = (W*H)/world_size;
    int inicial;
    int r = (W*H)%world_size;
    if(rank < r){
    	inicial =  rank*size + rank;
    	size++;
    }
    else inicial = rank*size + r;
	int *results = new int[size];
	#pragma omp parallel for num_threads(threads)
    for(int k = 0; k < size; k++){
    	int x = (k + inicial)/W;
		int y = (k + inicial)%H;
		// printf("%d %d    %d\n", x, y, n);
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
		// printf("%d\n", j);
		// printf("%d\n", j);

		results[k] = j;
		// printf("%d\n", j);
	}
    if(rank == 0){
		png::image< png::rgb_pixel > imagem(W, H);
		string saida = argv[9];
		const int M = 1000;
		int j; //para ficar parecido aos outros
		int m_size = size;
		for(int i = 0; i < world_size; i++){

			int m_inicial = i*(W*H)/world_size;
			if(i < r) m_inicial += i;
			else{
				m_inicial += r
				m_size = (W*H)/world_size;
			}
			if(i != 0) MPI_Recv(results, m_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			for(int k = 0; k < m_size; k++) {
			//Acho que o X e o Y é algo assim
			// int x = p/N;
			// int y = p%N;

				j = results[k];
                int x, y;
                x = (m_inicial + k)/H;
                y = (m_inicial + k)%W;
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
		imagem.write(saida);
	}
	else{
		MPI_Send(results, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	//printf("copiei\n");


	//Sempre bom desalocar
	delete [] results;
}

//////////////////////////////////////CUDA//////////////////////////////////////
//Função que furtei do add.cu
void cudaAssert(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("Erro %d!\n", err);
        abort();
    }
}


__global__ void gpu_calculation(float c0r, float c0i, float float_step, float imag_step, int *results, unsigned n, int W, int H, int inicial){

	// index = m*x + y
	const long unsigned globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

	// printf("%d %d\n", blockIdx.x, threadIdx.x);

	if (globalIndex < n) {
        //calcular os complexos na mão
		int x = (globalIndex + inicial)/W;
		int y = (globalIndex + inicial)%H;
		// printf("%d %d    %d\n", x, y, n);
        float point_r = c0r+x*float_step;
        float point_i = c0i+y*imag_step;

		// printf("%f %f\n", point_r, point_i);
    	const int M = 1000;

		// valor Zj que falhou
		// -1 se não tiver falhado
		int j = -1;

		//Valor da iteração passada
		float old_r = 0;
		float old_i = 0;
		float aux = 0;

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
	float c0r = stof(argv[1]);
	float c0i = stof(argv[2]);
	float c1r = stof(argv[3]);
	float c1i = stof(argv[4]);

	int W = stoi(argv[5]);
	int H = stoi(argv[6]);

	int threads = stoi(argv[8]);

	float float_step = (c1r - c0r)/W;
	float imag_step = (c1i - c0i)/H;

	//printf("step gpu %f %f %f %f\n", c1r, c1i, c0r, c0i);
	//printf("step gpu %f %f %f %f\n", float_step, imag_step,(c1r - c0r), (c1i - c0i));
	int world_size; //numero de processos
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank; //rank do processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = (W*H)/world_size; //tamanho do vetor que cada processo vai calcular
    int inicial = rank*size; //indice que representa o primeiro ponto que cada processo vai calcular
    int r = (W*H)%world_size; //arruma size e inicial para quantidades de pixels não divisivel pela quantidade de processos
    if(rank < r){
    	inicial += rank;
    	size++;
    }
    else inicial += r;
	const int NUM_BLOCKS = (size + threads-1)/threads;
	int *results = new int[size];
	int *cuda_results;
	cudaAssert(cudaMalloc(&cuda_results, size*sizeof(int)));
	gpu_calculation<<<NUM_BLOCKS, threads>>>(c0r, c0i, float_step, imag_step, cuda_results, size, W, H, inicial);
	cudaAssert(cudaMemcpy(results, cuda_results, size*sizeof(int), cudaMemcpyDeviceToHost));
	cudaFree(cuda_results);
    if(rank == 0){
		png::image< png::rgb_pixel > imagem(W, H);
		string saida = argv[9];
		const int M = 1000;
		int j; //para ficar parecido aos outros
		int m_size = size;
		for(int i = 0; i < world_size; i++){
			// recebe o vetor com os os resultados de se e quando convergiu
			int m_inicial = i*(W*H)/world_size;
			if(i < r) m_inicial += i;
			else{
				m_inicial += r
				m_size = (W*H)/world_size;
			}
			if(i != 0) MPI_Recv(results, m_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			for(int k = 0; k < m_size; k++) {
				//Acho que o X e o Y é algo assim
				// int x = p/N;
				// int y = p%N;

				j = results[k];
				//calcula o indice dos pixels baseado no indice do vetor de resultado
                int x, y;
                x = (m_inicial + k)/H;
                y = (m_inicial + k)%W;
				//preenche a imagem
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
		imagem.write(saida);
	}
	else{
		MPI_Send(results, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	//printf("copiei\n");


	//Sempre bom desalocar
	delete [] results;

}
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]){
	//processar os args
	//mbrot <C0_real> <C0_IMAG> <C1_real> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>
	MPI_Init(&argc, &argv);
	if(argc < 10){
		cout << "Incorrect Number of Args" << endl;
		cout << "Usage:" << endl;
		cout << "mbrot <C0_real> <C0_IMAG> <C1_real> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>" << endl;
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

    MPI_Finalize();
	return 0;

}
