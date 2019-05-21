
#include <iostream>
#include <string>
#include <complex>
#include "png++/png.hpp"

using namespace std;

template<class real> // para trocar entre float e double
void mandelbrot_seq(char *argv[]){
	real c0r = stod(argv[1]);
	real c0i = stod(argv[2]);
	real c1r = stod(argv[3]);
	real c1i = stod(argv[4]);

	int W = stoi(argv[5]);
	int H = stoi(argv[6]);

	string CPU_GPU = argv[7];

	int threads = stoi(argv[8]);

	string saida = argv[9];

	real real_step = (c1r - c0r)/W;
	real imag_step = (c1i - c0i)/H;

	png::image< png::rgb_pixel > imagem(W, H);

	for (png::uint_32 y = 0; y < imagem.get_height(); ++y){
		for (png::uint_32 x = 0; x < imagem.get_width(); ++x){
			complex<real> point ( c0r+x*real_step , c0i+y*imag_step);
			const int M = 1000;

			// valor Zj que falhou
			// -1 se não tiver falhado
			int j = -1;

			//Valor da iteração passada
			complex<real> old_num (0,0);

			//Calcula o mandebrot
			for(int i = 1; i <= M; i++){

				old_num = old_num*old_num + point;

				if( (abs(old_num) > 2 )){
					j = i;
					break;
				}
			}
			printf("%d\n", j);

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

	string CPU_GPU = argv[7];

	int threads = stoi(argv[8]);

	string saida = argv[9];

	real real_step = (c1r - c0r)/W;
	real imag_step = (c1i - c0i)/H;

	printf("step cpu %f %f %f %f\n", c1r, c1i, c0r, c0i);
	printf("step cpu %f %f %f %f\n", real_step, imag_step,(c1r - c0r)/W, (c1i - c0i)/H);
	printf("%d %d\n", W, H);

	png::image< png::rgb_pixel > imagem(W, H);
	png::uint_32 y;
	png::uint_32 x;

	#pragma omp parallel for collapse(2) num_threads(threads)
		for (y = 0; y < imagem.get_height(); ++y){
			for (x = 0; x < imagem.get_width(); ++x){
				complex<real> point ( c0r+x*real_step , c0i+y*imag_step);
				printf("%d %d\n",x,y );
				printf("%f %f\n", std::real(point), std::imag(point));
				const int M = 1000;

				// valor Zj que falhou
				// -1 se não tiver falhado
				int j = -1;

				//Valor da iteração passada
				complex<real> old_num (0,0);
				// if (x == 0 && y == 0) {
				// cout << "Real part: " << std::real(old_num) << endl;
				// cout << "Imaginary part: " << std::imag(old_num) << endl;
				// old_num *= old_num;
				// cout << "Real part: " << std::real(old_num) << endl;
				// cout << "Imaginary part: " << std::imag(old_num) << endl;
				// }

				//Calcula o mandebrot
				for(int i = 1; i <= M; i++){

					// cout << "Real part: " << std::real(old_num) << endl;
 					// cout << "Imaginary part: " << std::imag(old_num) << endl;
					// cout << "point Real part: " << std::real(point) << endl;
 					// cout << "point Imaginary part: " << std::imag(point) << endl;
					old_num = old_num*old_num + point;
					// cout << "new Real part: " << std::real(old_num) << endl;
					// cout << "new Imaginary part: " << std::imag(old_num) << endl;
					// cout << endl;

					if( (abs(old_num) > 2 )){
						j = i;
						break;
					}
				}
				// printf("%d\n", j);

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

int main(int argc, char *argv[]){
	//processar os args
	//mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>

	if(argc < 10){
		cout << "Incorrect Number of Args" << endl;
		cout << "Usage:" << endl;
		cout << "mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>" << endl;
		return 0;
	}

	string cgpu(argv[7]);
	if(cgpu == "cpu")
		mandelbrot_omp<float>(argv);
	else if(cgpu == "seq")
		mandelbrot_seq<float>(argv);
	else
		cout << "Errrooou";
	return 0;

}