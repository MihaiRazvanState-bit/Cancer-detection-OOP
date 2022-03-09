#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/hal/interface.h"
#include <iomanip>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <unistd.h> 
#include <stdio.h> 
#include <stdlib.h> 

using namespace cv;
using namespace std;

struct tumori {
	string nume,poza[10];
	int nr;
};

Mat convertBinary(Mat img, int threshold) {

	Mat binImg = Mat(img.rows, img.cols, img.type());

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);

			if (intensity.val[0] > threshold)
			{
				binImg.at<cv::Vec3b>(i, j).val[0] = 255;
				binImg.at<cv::Vec3b>(i, j).val[1] = 255;
				binImg.at<cv::Vec3b>(i, j).val[2] = 255;
			}
			else
			{
				binImg.at<cv::Vec3b>(i, j).val[0] = 0;
				binImg.at<cv::Vec3b>(i, j).val[1] = 0;
				binImg.at<cv::Vec3b>(i, j).val[2] = 0;
			}
		}

	return binImg;
}

Mat addImg(Mat highpassImg, Mat img) {

	for (int i = 1; i <= img.rows - 2; i++)
		for (int j = 1; j <= img.cols - 2; j++)
		{
			img.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0] + highpassImg.at<cv::Vec3b>(i, j)[0];
			img.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1] + highpassImg.at<cv::Vec3b>(i, j)[1];
			img.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2] + highpassImg.at<cv::Vec3b>(i, j)[2];

			if (img.at<cv::Vec3b>(i, j)[0] > 255)
			{
				img.at<cv::Vec3b>(i, j)[0] = 255;
				img.at<cv::Vec3b>(i, j)[1] = 255;
				img.at<cv::Vec3b>(i, j)[2] = 255;
			}

		}

	return img;
}

Mat toDouble(Mat img) {

	for (int i = 1; i <= img.rows - 2; i++)
		for (int j = 1; j <= img.cols - 2; j++)
		{
			img.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0];
			img.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1];
			img.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2];
		}

	return img;
}

Mat filter_custom(Mat img, Mat strel) {

	Mat tempImg = img.clone();
	for (int i = 1; i <= img.rows - 2; i++)
		for (int j = 1; j <= img.cols - 2; j++)
		{
			int sum = 0;
			int p = 0;
			for (int m = -((strel.rows - 1) / 2); m <= (strel.rows - 1) / 2; m++)
			{
				int q = 0;
				for (int n = -((strel.cols - 1) / 2); n <= (strel.cols - 1) / 2; n++)
				{
					sum = sum + ((int)img.at<cv::Vec3b>(i + m, j + n)[0] * (int)strel.at<int>(p, q));
					//tempImg.at<cv::Vec3b>(i+m,j+n)[1] = (int)img.at<cv::Vec3b>(i+m,j+n)[1] + (int)strel.at<int>(p,q);
					//tempImg.at<cv::Vec3b>(i+m,j+n)[2] = (int)img.at<cv::Vec3b>(i+m,j+n)[2] + (int)strel.at<int>(p,q);
					q++;
				}
				p++;
			}

			tempImg.at<cv::Vec3b>(i, j)[0] = sum;
			tempImg.at<cv::Vec3b>(i, j)[1] = sum;
			tempImg.at<cv::Vec3b>(i, j)[2] = sum;
		}

	return tempImg;
}

//init phase
Mat faza_initiala(Mat img, Mat strel) {

	Mat tempImg = Mat(img.rows, img.cols, img.type(), Scalar(0, 0, 0, 0));

	for (int i = 0; i <= img.rows - 1; i++)
		for (int j = 0; j <= img.cols - 1; j++)
		{
			int validFlag = true;
			for (int m = 0; m <= strel.rows - 1; m++)
				for (int n = 0; n <= strel.cols - 1; n++)
				{
					if ((int)img.at<cv::Vec3b>(i + m, j + n)[0] != (int)strel.at<int>(m, n))
					{
						validFlag = false;
					}
				}
			if (validFlag)
			{
				tempImg.at<Vec3b>(i, j)[0] = 255;
				tempImg.at<Vec3b>(i, j)[1] = 255;
				tempImg.at<Vec3b>(i, j)[2] = 255;
			}
		}

	return tempImg;
}

//functie predictie extindere tumora
Mat extindere_tum(Mat img, Mat strel) {

	Mat outImg = Mat(img.rows, img.cols, img.type(), Scalar(0, 0, 0, 0));
	for (int i = 0; i <= img.rows - 1; i++)
	{
		for (int j = 0; j <= img.cols - 1; j++)
		{
			if ((int)img.at<cv::Vec3b>(i, j)[0] == 255)
			{
				for (int m = 0; m <= strel.rows - 1; m++)
					for (int n = 0; n <= strel.cols - 1; n++)
					{
						outImg.at<cv::Vec3b>(i + m, j + n)[0] = (int)strel.at<int>(m, n);
						outImg.at<cv::Vec3b>(i + m, j + n)[1] = (int)strel.at<int>(m, n);
						outImg.at<cv::Vec3b>(i + m, j + n)[2] = (int)strel.at<int>(m, n);
					}
			}
		}
	}
	return outImg;
}

void label(Mat& img, int i, int j, int r, int g, int b) {

	cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);
	int o_r = intensity.val[0];
	if (i < img.rows && j < img.cols && i >= 0 && j >= 0 && o_r == 255)
	{
		cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);

		intensity.val[0] = r;
		intensity.val[1] = g;
		intensity.val[2] = b;

		img.at<cv::Vec3b>(i, j) = intensity;

		label(img, i + 1, j, r, g, b);
		label(img, i, j + 1, r, g, b);
		label(img, i, j - 1, r, g, b);
		label(img, i - 1, j, r, g, b);
	}
}

//functie gasire tumora metoda celor 4 componente
Mat fourConn(Mat& img) {

	int r = 0, g = 0, b = 255;
	int cells = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);
			int o_r = intensity.val[0];
			int o_g = intensity.val[1];
			int o_b = intensity.val[2];

			if (o_r == 255)
			{
				label(img, i, j, r, g, b);
				r = r + 0;
				g = g + 0;
				b = b + 0;

				cells++;
			}
		}

	cout << "\nNumarul tumorilor gasite = " << cells << endl;

	return img;
}

string Hamming_d(Mat img1, Mat img2){
	
	string c="";
	
	for(int i=0;i < min(img1.rows-1,img2.rows-1); i++){
		for(int j=0; j < min(img1.cols-1,img2.cols-1) ;j++){
			if(img1.at<cv::Vec3b>(i,j)==img2.at<cv::Vec3b>(i,j))
				c = c + "0";
			else
				c = c + "1";
		}
	}
	return c;
}

int directie(Mat img1,Mat img2){
	int d=0;
	for(int i=0;i < min(img1.rows-1,img2.rows-1); i++)
		for(int j=0; j < min(img1.cols-1,img2.cols-1) ;j++){
			cv::Vec3b intensity1 = img1.at<cv::Vec3b>(i, j);
			cv::Vec3b intensity2 = img2.at<cv::Vec3b>(i, j);
			if(intensity1.val[0]>intensity2.val[0])
				d++;
			else if(intensity1.val[0]<intensity2.val[0])
				d--;
		}
	return d;
}


double d_Levenstein( const Mat A, const Mat B ) {
	if ( A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols ) {
		// Calculate the L2 relative error between images.
		double errorL2 = norm( A, B, CV_L2 );
		// Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
		double similarity = errorL2 / (double)( A.rows * A.cols );
		return similarity;
	}
	else {
		//Images have a different size
		return 100000000.0;  // Return a bad value
	}
}


int main(){
	int i, j, n, status;    
	tumori t[100];
	fstream f1;
	ifstream f;
	f.open("Date_intrare.txt");
	f >> n;
	for (i = 0; i < n; i++) {
		f >> t[i].nume;
		f >> t[i].nr;
		for (j = 0; j < t[i].nr; j++)
			f >> t[i].poza[j];
		}
	f.close();
	Mat img_precedenta;
	for (i = 0; i < n; i++) {
		string path = t[i].nume;//fiecare nume din date_intrare.txt
		status = mkdir(path.c_str(),0777);//creare folder conform date_intrare.txt
		int dir=0,b[100],nr=0,k;
		for (j = 0; j < t[i].nr; j++) {
			//citirea intrare
			Mat img = imread(t[i].poza[j]);
			//waitKey(0);

			//extragere nume separat pentru lucru cu imagini
			string s1(t[i].poza[j]);
			size_t pos = s1.find(".");
			string addr = s1.substr(0, pos);

			if(j!=0){
				f1.open(path+"/"+(addr+"Rez_Hamming.txt"),ios::out);
				f1.close();
				f1.open(path+"/"+(addr+"Rez_Levenstein.txt"),ios::out);
				f1.close();
			}

			//folosit in functia hamming
			cout << t[i].nume;

			//aplicarea filtrului grayscale
			Mat gray_image(img.size(), CV_8UC1);
			cvtColor(img, gray_image, CV_BGR2GRAY);
			//imshow(addr+"-Grayscale", gray_image);
			imwrite(path+"/"+addr+"-grayscale.bmp", gray_image);
			//waitKey(0);

			//reducerea dil fct pred
			Mat kern = (Mat_<int>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
			Mat highpassImg;
			filter2D(img, highpassImg, img.depth(), kern);

			highpassImg = addImg(highpassImg, img);
			//imshow(addr+"-Copy Original", highpassImg);
			imwrite(path+"/"+addr+"-original.bmp", highpassImg);
			//waitKey(0);

			//aplicarea functiei contur
			Mat binImg = convertBinary(highpassImg, 160);
			//imshow(addr+"-Imaginea Binara", binImg);
			imwrite(path+"/"+addr+"-Imagine_binara.bmp", binImg);
			//waitKey(0);

			//aplicarea operatiunilor logice pt reteaua neuronala
			Mat strel = (Mat_<int>(15, 15) << 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
				);

			//faza 1
			Mat outImg = faza_initiala(binImg, strel);
			//imshow(addr+"-Faza initiala", outImg);
			imwrite(path+"/"+addr+"-Imagine_tum_init.bmp", outImg);
			//waitKey(0);

			//predictie extindere
			outImg = extindere_tum(outImg, strel);
			//imshow(addr+"-Dupa extindere", outImg);
			imwrite(path+"/"+addr+"-Imagine_tumor_extinsa.bmp", outImg);
			//waitKey(0);

			//gasirea tumorei
			outImg = fourConn(outImg);
			//imshow(addr+"-Componenta marcata", outImg);
			imwrite(path+"/"+addr+"-Componenta_Marcata.bmp", outImg);
			//waitKey(0);



			if( j != 0 ){
				f1.open(path+"/"+addr+"Rez_Hamming.txt",ios::app);
				f1 << Hamming_d(img_precedenta, binImg);
				f1.close();
				f1.open(path+"/"+addr+"Rez_Levenstein.txt",ios::app);
				f1 << d_Levenstein(img_precedenta, binImg);
				f1.close();
				b[nr]=directie(img_precedenta,binImg);
				dir+=b[nr++];
			}
			img_precedenta = binImg;

		}
		f1.open(path+"/"+t[i].nume+"Evolutie.txt",ios::out);
		for(k=0;k<nr;k++)
			f1<<k+1<<"->"<<k+2<<"   ";
		f1<<endl;
		for(k=0;k<nr;k++)
			f1<<setw(4)<<b[k]<<" ";
		f1<<endl;
		if(dir<0)
			f1<<"Directia de evolutie a tumorii este spre scadere. "<<endl;
		else if(dir>0)
			f1<<"Directia de evolutie a tumorii este spre crestere. "<<endl;
		else
			f1<<"Directia de evolutie a tumorii este spre stagnare. "<<endl;
		f1.close();
	}
	cout << "\n\nTerminare program...";
	getchar();

    return 0;
}