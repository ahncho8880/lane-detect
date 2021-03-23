#include<iostream>
#include"opencv2/opencv.hpp"
#include"CurrentLane.h"
#include"CLine.h"
#include"LaneDetect.h"

#define MIN(a,b) ((a)>(b)?(b):(a))
using namespace cv;
using namespace std;
typedef struct PolarPoint {
    float rho;
    float theta;
} PolarPoint;

int main()
{
   

    cout << "***** Lane Detectection Program *** by. JY * JH * CH *****" << endl;
    //cout << "File name: ";
    //char filename[50];
    //cin >> filename;
    //VideoCapture capture(filename);
    VideoCapture capture("/Users/ahnchoru/Desktop/storage/line_detect_2017/test4.mp4");
    LaneDetect lanedetect;
    if (!capture.isOpened())
    {
        cout << "Can not open capture !!!" << endl;
        return 0;
    }

    
    /*ø¯øµªÛ¿« ≈©±‚*/
    int width, height;
    width = (int)capture.get(CAP_PROP_FRAME_WIDTH);
    height = (int)capture.get(CAP_PROP_FRAME_HEIGHT);
    Size size = Size((int)width, (int)height);

    cout << "size: " << size << endl;

    /*ø¯øµªÛ¿« fps(√ ¥Á «¡∑π¿”ºˆ)*/
    int fps = (int)(capture.get(CAP_PROP_FPS));

    cout << "fps: " << fps << endl;

    int frameNum = -1;
    Mat frame, roiframe, roiframe_L, roiframe_R;
    int roi_x = 0, roi_y = height / 5 * 3; //∞¸Ω…øµø™ Ω√¿€¡°¿« ¡¬«• x,y
                                       //width = 1280;
                                       //height = 720/2=340;
                                       /*¿ßø°º≠ 1/3¡ˆ¡°∫Œ≈Õ æ∆∑° 1/2¿« øµø™¿ª ROI∑Œ º≥¡§*/
    
    /*∞¸Ω…øµø™ ¡§¿«*/
    //Parameter: Ω√¿€¡°¿«(x,y), ∞¸Ω…øµø™¿« ≈©±‚
    Rect roi(roi_x, roi_y, width, height / 6);
    Rect roileft(roi_x, roi_y, width / 2, height / 6);
    Rect roiright((width - roi_x) / 2, roi_y, width / 2, height / 6);

    CurrentLane *curlane;
    double smallA = 150;
    Point crx;
    Point cry;
    Point crv;
    int seq_ey_R = 0;
    int seq_ey_L = 0;
    int check = 0;
    PolarPoint LLnMovAvg = { 0 }, RLnMovAvg = { 0 }, LLnPt, RLnPt;
    while (char(waitKey(1)) != 'q' && capture.isOpened())
    {
        capture >> frame;
        if (frame.empty())
            break;
        cout << "Detect CurrentLane..." << endl;
        


        /*∞¸Ω…øµø™ º≥¡§*/
        roiframe = frame(roi);
        roiframe_L = frame(roileft);
        roiframe_R = frame(roiright);

        Mat preImage, edgeImage;
        /*¿¸√≥∏Æ*/
        preImage= lanedetect.preprocessing(frame,roi,roileft,roiright);
        Canny(preImage, edgeImage, 100, 210, 3);
        

        /*øß¡ˆ¿ÃπÃ¡ˆ øµø™º≥¡§*/
        Mat roiEdge_R, roiEdge_L;
        roiEdge_R = edgeImage(roiright);
        roiEdge_L = edgeImage(roileft);
        
        /*¡¬øÏøµø™¿ª ∂Û∫ß∏µ*/
        //∂Û∫ß∏µø° « ø‰«— ∫Øºˆ
        Mat img_labels_R, stats_R, centroids_R;
        Mat img_labels_L, stats_L, centroids_L;
        
        /*¡¬øÏ Labelled øµø™¿« ∞≥ºˆ*/
        int numOfLabels_R = connectedComponentsWithStats(roiEdge_R, img_labels_R, stats_R, centroids_R, 8, CV_32S);
        int numOfLabels_L = connectedComponentsWithStats(roiEdge_L, img_labels_L, stats_L, centroids_L, 8, CV_32S);

        // Labelled øµø™ø°º≠ ¡˜º±¿ª √ﬂ√‚«ÿ ¿˙¿Â«œ±‚ ¿ß«— ∏ﬁ∏∏Æ «“¥Á (√÷¥Î label ∞≥ºˆ∏∏≈≠)
        CLine*lines_R = (CLine *)malloc(sizeof(CLine)*numOfLabels_R);
        CLine* lines_L = (CLine *)malloc(sizeof(CLine)*numOfLabels_L);


        //∂Û∫ßµ» ∞Õ¡ﬂ øﬁ¬ øµø™¿« º± √ﬂ√‚«ÿº≠ ¡°∞˙ index∏¶ ¿˙¿Â
        int left_lines = lanedetect.extractLine_L(frame,roiframe, roiframe_L, lines_L, numOfLabels_L, roiEdge_L, img_labels_L, stats_L, centroids_L);
        int right_lines = lanedetect.extractLine_R(frame,roiframe, roiframe_R, lines_R, numOfLabels_R, roiEdge_R, img_labels_R, stats_R, centroids_R);
        
        lanedetect.getCurrentlane(frame, &smallA, crv, crx, cry, seq_ey_R, seq_ey_L, lines_R, lines_L, right_lines, left_lines, &check, width / 2, roi_y);

        
        //CLine* curlines_L = (CLine *)malloc(sizeof(CLine)*numOfLabels_L);
        //lanedetect.currentLine_L(frame, q, curlines_L, numOfLabels_L, stats_L, centroids_L);
        //lanedetect.currentLane(frame, &smallA, crv, crx, cry, lines_R, lines_L, right_lines, left_lines, &check, width/2, roi_y);
        /*√ﬂ√‚µ» ¡˜º±¿ª »≠∏Èø° √‚∑¬*/
        //lanedetect.displayLineinfo(frame, lines_R, right_lines, Scalar(0, 0, 255), Scalar(200, 200, 255), width / 2, roi_y);
 //        lanedetect.displayLineinfo(frame, lines_L, left_lines, Scalar(0, 0, 255), Scalar(200, 200, 255), 0, roi_y);
        //lanedetect.currentLine_L(frame,&linesinfo_L,lines_L, curlines_L, numOfLabels_L, stats_L, centroids_L);

        //µø¿˚«“¥Á ∏ﬁ∏∏Æ «ÿ¡¶
        free(lines_R);
        free(lines_L);
        //«ˆ¿Á ¬˜º± «•Ω√
        if (check >= 0) {
            vector<Point> Lane;
            vector<Point> fillLane;

            Lane.push_back(crv);
            Lane.push_back(crx);
            Lane.push_back(cry);

            Mat temp(roiframe.rows, roiframe.cols, CV_8UC1);
            for (int i = 0; i<temp.cols; i++)
                for (int j = 0; j<temp.rows; j++)
                    temp.at<uchar>(Point(i, j)) = 0;
        
            Mat curlane(roiframe.rows, roiframe.cols, CV_8UC3);
            Mat curarea(roiframe.rows, roiframe.cols, CV_8UC3);

            line(temp, crx, crv, 255, 3);
            line(temp, cry, crv, 255, 3);
            approxPolyDP(Lane, fillLane, 1.0, true);

            fillConvexPoly(temp, &fillLane[0], fillLane.size(), 255, 8, 0);
    
            fillConvexPoly(curlane, &fillLane[0], fillLane.size(), Scalar(255,255,0), 8, 0);

            addWeighted(roiframe, 0.7,curlane,0.2, 0.0, roiframe);
        }
        /*∞¸Ω…øµø™ «•Ω√*/
        (frame, roiright, Scalar(255, 255, 255), 3);
        (frame, roileft, Scalar(255, 255, 255), 3);
    
        /*Ω««‡ √¢*/
        namedWindow("Lane Detection", 0);
        imshow("Lane Detection", frame);
        
//        imshow("edge", edgeImage);
    }
    return 0;
}
