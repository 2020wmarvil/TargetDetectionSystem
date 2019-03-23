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

/* 		NOTES
	Dense vs Sparse optical flow
		dense (farnerback's algorithm)
		   - requires the use of every pixel in every computation 
		   - slow but accurate
		sparse (lucas-kanade algorithm)
		   - only requires a certain subset of pixels to perform a computation 
		   - fast but potentially inaccurate
*/

int main() {
	VideoCapture cap(1);

	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 20, 20);

	Mat old_frame, old_gray;
	std::vector<Point> old_corners;

	cap >> old_frame;
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

	goodFeaturesToTrack(old_gray, old_corners, 100, 0.3, 7);

	Mat frame, gray;
	std::vector<Point> corners;
	while (true) {
		cap >> frame;

		if (frame.empty()) break;

		cvtColor(frame, gray, COLOR_BGR2GRAY);

		Mat status, err;
		calcOpticalFlowPyrLK(old_gray, gray, old_corners, corners, status, err);

		gray.copyTo(old_gray);
		old_corners = corners;

		imshow(WIN_SRC, frame);

		if (char key = waitKey(30) == 27) break;
	}

	destroyAllWindows();

	return 0;
}
