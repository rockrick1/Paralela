
#include <iostream>
#include <string>
#include <complex>
#include "png++/png.hpp"

using namespace std;

//calcula o mandelbrot para um ponto
template<class real> //recebe do mandelbrot
inline void calculate_mandelbrot(complex<real> num, int x, int y, png::image< png::rgb_pixel > imagem){
	const int M = 1000;

	// valor Zj que falhou
	// -1 se não tiver falhado
	int j = -1;

	//Valor da iteração passada
	complex<real> old_num (0,0);

	//Calcula o mandebrot
	for(int i = 1; i <= M; i++){

		old_num = old_num*old_num + num;

		if( (abs(old_num) > 2 )){
			j = i;
			break;
		}
	}


	if (j == -1){
		imagem.set_pixel(x, y, png::rgb_pixel(0, 0, 0));
	}
	else{
		imagem.set_pixel(x, y, png::rgb_pixel(255, 255, 255));
	}

}

template<class real> // para trocar entre float e double
void mandelbrot(char *argv[]){
	////////////////////////////////// TESTES //////////////////////////////////
	//Teste Bom
	//./mandelbrot 0.27085 0.004640 0.27100 0.004810 1000 1000 cpu 2 mb.png
	complex<real> c0 (0.27085,0.004640);
	complex<real> c1 (0.27100,0.004810);

	int W = 1000;
	int H = 1000;

	string CPU_GPU = "cpu";

	int threads = 2;

	string saida = "mb.png";
	/////////////////////////////// real oficial ///////////////////////////////
	// complex<real> c0 (stod(argv[1]), stod(argv[2]));
	// complex<real> c1 (stod(argv[3]), stod(argv[4]));
	//
	// int W = stoi(argv[5]);
	// int H = stoi(argv[6]);
	//
	// string CPU_GPU = argv[7];
	//
	// int threads = stoi(argv[8]);
	//
	// string saida = argv[9];
	////////////////////////////////////////////////////////////////////////////

	real real_step = (c1.real() - c0.real())/W;
	real imag_step = (c1.imag() - c0.imag())/H;

	png::image< png::rgb_pixel > imagem(W, H);

	for (png::uint_32 y = 0; y < imagem.get_height(); ++y){
		for (png::uint_32 x = 0; x < imagem.get_width(); ++x){
			complex<real> point ( c0.real()+x*real_step , c0.imag()+y*imag_step);
			calculate_mandelbrot<real>(point, x, y, imagem);
		}
	}

	imagem.write(saida);
}

int main(int argc, char *argv[]){
	//processar os args
	//mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>

	// if(argc < 10){
	// 	cout << "Incorrect Number of Args" << endl;
	// 	cout << "Usage:" << endl;
	// 	cout << "mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>" << endl;
	// 	return 0;
	// }

	mandelbrot<float>(argv);
	return 0;

}