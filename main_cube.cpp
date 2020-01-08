#include "cv.hpp"
#include <iostream>
#include <fstream>

#include "system.h"
#include "CamModelGeneral.h"

//Extern Global Parameter for Print Result
vector<float> avgRate_vec;
vector<int> avgN_vec;
vector<int> avgGoodMatch_vec;

vector<double> TIME_avgGoodMatch_vec;
vector<double> TIME_avgDetect_vec;
vector<double> TIME_avgCompute_vec;


using namespace std;
using namespace cv;


vector<Mat> read_cube_imgs(int start_flag, int end_img_cnt);

/* main */
int main()
{
	int start_F_idx = 400, end_F_idx = 411;
	//cout << "SCAN START/END FRAME Idx: ";
	//cin >> start_F_idx;
	//cin >> end_F_idx;

	/* read data */
	vector<Mat> img_set = read_cube_imgs(start_F_idx, end_F_idx);	///if 0 <- get all the every img set

	Mat mask = imread("../../dataset/mask/loop2_front_mask.png", IMREAD_GRAYSCALE);
	Mat cube_mask = imread("../../dataset/cube_mask/gray_cubemap_front_mask_650.png", IMREAD_GRAYSCALE);
	if (cube_mask.empty() || mask.empty())
	{
		std::cout << "fail to read the mask: " << std::endl;
		exit(0);
	}

	std::string settingFilePath("../../Config/front_cam_params.yaml");

	//0: ORB, 1:BRISK, 2:AKAZE 3:ORB_EX
	F_test::System F_system(F_test::System::OMNI, F_test::System::ORB_EX, settingFilePath, img_set.size(), 3000);


	//Make cube img_set
	vector<Mat> cube_img_set;
	for (int ni = 0; ni < img_set.size(); ni++)
	{
		int offset = 0;
		int width = CamModelGeneral::GetCamera()->GetCubeFaceWidth(), height = CamModelGeneral::GetCamera()->GetCubeFaceHeight();
		cv::Mat cubemapImg(height * 3, width * 3, CV_8U, cv::Scalar::all(0));
		cv::Mat cubemapImg_front = cubemapImg.rowRange(height, 2 * height).colRange(width, 2 * width);
		cv::Mat cubemapImg_left = cubemapImg.rowRange(height, 2 * height).colRange(0 + offset, width + offset);
		cv::Mat cubemapImg_right = cubemapImg.rowRange(height, 2 * height).colRange(2 * width - offset, 3 * width - offset);
		cv::Mat cubemapImg_upper = cubemapImg.rowRange(0 + offset, height + offset).colRange(width, 2 * width);
		cv::Mat cubemapImg_lower = cubemapImg.rowRange(2 * height - offset, 3 * height - offset).colRange(width, 2 * width);

		//F_system.CvtFisheyeToCubeMap_reverseQuery_withInterpolation(cubemapImg, img_set[ni], cv::INTER_LINEAR);
		F_system.CvtFisheyeToCubeMap_reverseQuery(cubemapImg, img_set[ni]);
		//circle(cubemapImg, Point(450, 450), 4, Scalar(0, 0, 255), -1);

		cube_img_set.push_back(cubemapImg);
		//imshow("cube", cube_img_set[ni]);
		//waitKey(0);
	}







	F_system.TwoViewTest(cube_img_set, mask, cube_mask, 5, 33);//(img term), (20)
	F_system.print_RESULT();


	waitKey(0);

	return 0;
}
/* main */




vector<Mat> read_cube_imgs(int start_flag, int end_img_cnt)
{
	std::string fisheyeImgPath = "../../dataset/loop2_front/";
	std::ifstream fin("../../dataset/loop2_front/front_images.lst");

	// Retrieve paths to images
	std::vector<std::string> fisheyeImgNames;
	std::vector<double> vTimestamps;
	std::string line;
	while (std::getline(fin, line))
	{
		fisheyeImgNames.push_back(line);
		size_t p = line.find_last_of("_");
		std::stringstream ss(line.substr(0, p));
		double ts;
		ss >> ts;
		vTimestamps.push_back(ts);
	}

	// Main loop
	vector<Mat> img_set;
	cv::Mat im;
	int ni = 0;
	int end_flag = 0;

	if (end_img_cnt == 0)
		end_flag = fisheyeImgNames.size();
	else
		end_flag = end_img_cnt;

	ni = start_flag;
	while (true)
	{
		if (ni == end_flag)
			break;

		std::string fisheyeImgName = fisheyeImgPath + fisheyeImgNames[ni];
		im = cv::imread(fisheyeImgName, 0);
		img_set.push_back(im);

		if (im.empty())
		{
			cerr << endl
				<< "Failed to read camera" << endl;
			break;
		}
		if (ni == fisheyeImgNames.size())
		{
			cout << "frame end" << endl;
			break;
		}
		ni++;
	}

	const int imageNameCnt = fisheyeImgNames.size();
	std::cout << "ImageNameCnt find " << imageNameCnt << " image names" << std::endl;
	const int imageCnt = img_set.size();
	std::cout << "ImageCnt find " << imageCnt << " images" << std::endl;
	std::cout << "START/END FRAME Idx: [" << start_flag << " , " << end_img_cnt << "]" << std::endl;


	return img_set;
}



