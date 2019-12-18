#include "Frame.h"
#include <time.h>

extern vector<double> TIME_avgDetect_vec;
extern vector<double> TIME_avgCompute_vec;

namespace F_test
{
	//float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;


	Frame::Frame() {}
	Frame::Frame(const Mat img, const Mat mask, int feature_type)
	{
		F_img = img.clone();


		switch (feature_type)
		{
		case 0://ORB
			Extract_ORB(img, mask);///
			break;
		case 1://BRISK
			Extract_BRISK(img, mask);///
			break;
		case 2://AKAZE
			Extract_AKAZE(img, mask);///
			break;

		default:
			cerr << "!!! No adequate feature_type err !!!" << endl;
			exit(0);
			break;
		}

		N = mvKeys.size();
	}

	Frame::Frame(const Mat img, const Mat mask, ORBextractor* ORBextract)
	{
		F_img = img.clone();

		mpORBextractor = ORBextract;

		// ORB extraction
		Extract_ORB_EX(img, mask);

		N = mvKeys.size();
	}


	/*
	https://docs.opencv.org/3.4.1/db/d95/classcv_1_1ORB.html
	*/
	void Frame::Extract_ORB(Mat img, Mat mask)
	{
		Ptr<ORB> orbF = ORB::create(3000);
		//orbF->detectAndCompute(img, mask, mvKeys, mDescriptors);

		clock_t start_detect = clock();
		orbF->detect(img, mvKeys, mask);///
		double duration_detect = (double)(clock() - start_detect);

		clock_t start_compute = clock();
		orbF->compute(img, mvKeys, mDescriptors);///
		double duration_compute = (double)(clock() - start_compute);

		TIME_avgDetect_vec.push_back(duration_detect);
		TIME_avgCompute_vec.push_back(duration_compute);
	}

	/*
	https://docs.opencv.org/3.4.1/de/dbf/classcv_1_1BRISK.html
	*/
	void Frame::Extract_BRISK(Mat img, Mat mask)
	{
		Ptr<BRISK> briskF = BRISK::create();
		//briskF->detectAndCompute(img, mask, mvKeys, mDescriptors);

		clock_t start_detect = clock();
		briskF->detect(img, mvKeys, mask);///
		double duration_detect = (double)(clock() - start_detect);

		clock_t start_compute = clock();
		briskF->compute(img, mvKeys, mDescriptors);///
		double duration_compute = (double)(clock() - start_compute);

		TIME_avgDetect_vec.push_back(duration_detect);
		TIME_avgCompute_vec.push_back(duration_compute);
	}

	/*
	https://docs.opencv.org/3.4.1/d8/d30/classcv_1_1AKAZE.html
	*/
	void Frame::Extract_AKAZE(Mat img, Mat mask)
	{

		Ptr<AKAZE> akazeF = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT);//DESCRIPTOR_MLDB_UPRIGHT	DESCRIPTOR_MLDB
		//akazeF->detectAndCompute(img, mask, mvKeys, mDescriptors);

		clock_t start_detect = clock();
		akazeF->detect(img, mvKeys, mask);///
		double duration_detect = (double)(clock() - start_detect);

		clock_t start_compute = clock();
		akazeF->compute(img, mvKeys, mDescriptors);
		double duration_compute = (double)(clock() - start_compute);

		TIME_avgDetect_vec.push_back(duration_detect);///
		TIME_avgCompute_vec.push_back(duration_compute);
	}

	void Frame::Extract_ORB_EX(Mat img, Mat mask)
	{

		if (img.channels() != 1)
			cvtColor(img, img, CV_BGR2GRAY);

		///detection part
		clock_t start_detect = clock();

		// ORB extraction
		(*mpORBextractor)(img, mask, mvKeys, mDescriptors);

		double duration_detect = (double)(clock() - start_detect);

		///destriptor part
		clock_t start_compute = clock();


		double duration_compute = (double)(clock() - start_compute);

		TIME_avgDetect_vec.push_back(duration_detect);///
		TIME_avgCompute_vec.push_back(duration_compute);
	}


}
