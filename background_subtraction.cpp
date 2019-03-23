#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/features2d.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

string WIN_SRC = "Camera";
string WIN_MOG2 = "MOG2";

const int thresh = 50;

int main()
{
	VideoCapture cap(1);

	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 20, 20);
	namedWindow(WIN_MOG2, WINDOW_AUTOSIZE);
	moveWindow(WIN_MOG2, 820, 20);

	Ptr<BackgroundSubtractor> mog2 = createBackgroundSubtractorMOG2();

	Mat frame, mog2Mask, mog2Thresh;
	while (true)
	{
		cap >> frame;

		if (frame.empty())
			break;

		mog2->apply(frame, mog2Mask);

		threshold(mog2Mask, mog2Thresh, thresh, 255, THRESH_TOZERO);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(mog2Mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		double max_area = 0, max_i = -1;
		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			if (area > max_area)
			{
				max_area = area;
				max_i = i;
			}
		}

		if (max_i != -1)
		{
			drawContours(frame, contours, max_i, Scalar(0, 0, 255), 2, LINE_8, hierarchy, 0);
			rectangle(frame, boundingRect(contours[max_i]), Scalar(0, 255, 0));
		}

		imshow(WIN_SRC, frame);
		imshow(WIN_MOG2, mog2Mask);

		if (char key = waitKey(30) == 27)
		{
			break;
		}
	}

	return 0;
}
