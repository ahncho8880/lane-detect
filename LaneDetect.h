#pragma once
#ifndef _LaneDetect_H_
#define _LaneDetect_H_
#include"CLine.h"

typedef struct LinesInfo {
    vector<Vec4i> blob_lines;    //º±¿ª ∞®¡ˆ«œ±‚ ¿ß«— ∏∂¡ˆ∏∑ ¡°¿ª ∆˜«‘«— ∫§≈Õ
    int line_index;
}LinesInfo;
class curPoint {
public:
    Point end;
    Point start;
};
class LaneDetect {
public:
    Mat preprocessing(Mat frame, Rect roi, Rect roileft, Rect roiright);
    void displayLineinfo(Mat img, CLine * lines, int num_lines, Scalar linecolor, Scalar captioncolor, int width, int height);
    void detectcolor(Mat& image, double minH, double maxH, double minS, double maxS, Mat& mask);
    double getAngle(Point a, Point b, Point c);
    //void currentLane(Mat& image, double* angle, Point &curv, Point &curx, Point &cury, CLine* lines_R, CLine* lines_L, int right_lines, int left_lines, int *check, int width, int height);
    int extractLine_L(Mat &frame, Mat &roiframe, Mat &frame_L, CLine * lines, int num_labels, Mat edge_img, Mat &img_labels, Mat stats, Mat centroids);
    int extractLine_R(Mat &frame, Mat &roiframe, Mat &frame_R, CLine * lines, int num_labels, Mat edge_img, Mat &img_labels, Mat stats, Mat centroids);
    void getCurrentlane(Mat & image, double* angle, Point & curv, Point & curx, Point & cury, int seq_ey_R, int seq_ey_L, CLine * lines_R, CLine * lines_L, int right_lines, int left_lines, int * check, int width, int height);
    void drawlines(Mat& frame, Mat &img_labels);
};
#endif
