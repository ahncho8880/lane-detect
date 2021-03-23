#include<iostream>
#include <algorithm>
#include"opencv2/opencv.hpp"
#include"CurrentLane.h"
#include"CLine.h"
#include"LaneDetect.h"

#define MIN_AREA_PIXELS 300
#define LINE_SPACING_PIXELS 30

using namespace cv;
using namespace std;

Mat LaneDetect::preprocessing(Mat frame, Rect roi, Rect roileft, Rect roiright) {
    /*¿¸√≥∏Æ*/
    Mat    grayImage, otsuImage, closedImage, blurImage;

    cvtColor(frame, grayImage, COLOR_BGR2GRAY);    //±◊∑π¿ÃΩ∫ƒ…¿œ∑Œ ∫Ø»Ø
    adaptiveThreshold(grayImage, otsuImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 81, -50);//¿˚¿¿«¸threshold
    Mat morphfillter(4, 4, CV_8U, Scalar(1)); //ø¨ªÍ« ≈Õ
    morphologyEx(otsuImage, closedImage, MORPH_CLOSE, morphfillter);    //≈¨∑Œ¬°ø¨ªÍ
    medianBlur(closedImage, blurImage, 3);

    return blurImage;
}
void LaneDetect::drawlines(Mat &frame, Mat &img_labels) {
    for (int q = 0; q < img_labels.rows; ++q) {
        int *label_L = img_labels.ptr<int>(q); //label º˝¿⁄
        Vec3b* pixel = frame.ptr<Vec3b>(q); //∏Ò¿˚øµªÛ¿« «»ºø
        for (int w = 0; w < img_labels.cols; w++) {
            if (label_L[w] == 3 || label_L[w] == 1 || label_L[w] == 2) {
                pixel[w][2] = 255;
                pixel[w][1] = 0;
                pixel[w][0] = 0;

            }
        }
    }

}
bool lessY(curPoint a, curPoint b) {
    return a.end.y < b.end.y;
}

int LaneDetect::extractLine_L(Mat &frame, Mat &roiframe, Mat &frame_L, CLine * lines, int num_labels, Mat edge_img, Mat &img_labels, Mat stats, Mat centroids)
{
    vector<Vec4i> blob_lines;    //º±¿ª ∞®¡ˆ«œ±‚ ¿ß«— ∏∂¡ˆ∏∑ ¡°¿ª ∆˜«‘«— ∫§≈Õ
    int line_index = 0;
    curPoint cur;
    vector<curPoint> yseq;
    
    for (int i = 1; i < num_labels; i++)
    {
        // blob area∞° ≥ π´ ¿€¿∏∏È π´Ω√
        if (stats.at<int>(i, CC_STAT_AREA) < MIN_AREA_PIXELS) continue;
        
        //Houghtrasform »ƒ line ¡∂∞¢ √ﬂ√‚
        HoughLinesP(edge_img, blob_lines, 1, CV_PI / 180, 100, 100, 2);
        //∞À√‚«— º± ±◊∏Æ±‚
        drawlines(frame_L, img_labels);
        
        //line¿ª «œ≥™µµ √ﬂ√‚ ∏¯«ﬂ¿ª ∞ÊøÏ ¿Ã»ƒ∏¶ π´Ω√«œ∞Ì ¥Ÿ¿Ω blob¿∏∑Œ ≥—æÓ∞®
        if (blob_lines.size() == 0) continue;


        double longDistance = 0;
        int longDistanceIndex = 0;
        
        //√ﬂ√‚«— line ¡∂∞¢
        for (int k = 0; k < blob_lines.size(); k++)
        {
            Vec4i L = blob_lines[k];
            double distance = (L[2] - L[0])*(L[2] - L[0]) + (L[3] - L[1])*(L[3] - L[1]);
            if (distance > longDistance && distance > LINE_SPACING_PIXELS * 2) {
                longDistance = distance;
                longDistanceIndex = k;
            }

            // ¿ÃπÃ¡ˆ blob¿« center∏¶ ±‚¡ÿ¿∏∑Œ ªı∑ŒøÓ line ª˝º∫


            double rate = ((double)(L[3] - L[1])) / ((double)(L[2] - L[0]) + 0.5);
            //printf("%lf\n",rate);
            if (rate > -0.1 && rate < -0.5) break; //øﬁ¬ øµø™¿« º±¿« ±‚øÔ±‚ ¡∂∞«ø° ∫Œ«’«œ¡ˆ æ ¿∏∏È break

            // ∏µÁ ¬˜º± ¿˙¿Â
            int unique_line_flag = 1;

            // ¿ÃπÃ ª˝º∫«— line∞˙ ¡ﬂ∫πµ«¥¬(≥ π´ ∞°±ÓøÓ) line¿∫ ªı∑Œ ª˝º∫«œ¡ˆ æ ¥¬¥Ÿ
            for (int j =0; j < line_index; j++)
            {
                if (lines[j].dist_to_point(Point(L[0], L[1])) < LINE_SPACING_PIXELS)
                    unique_line_flag = 0;
            }

            // ªı∑ŒøÓ line¿Œ ∞ÊøÏ ª˝º∫«œø© lines ∏ﬁ∏∏Æ øµø™ø° ¿˙¿Â
            if (unique_line_flag)
            {
                
                double centerx = centroids.at<double>(i, 0);
                double centery = centroids.at<double>(i, 1);

                int left = stats.at<int>(i, CC_STAT_LEFT);
                int width = stats.at<int>(i, CC_STAT_WIDTH);
                
                lines[line_index] = CLine(rate, Point((int)centerx, (int)centery), left, width);
                
                //∞À√‚«— º±¿∫ ¡° (L[0],L[1])∞˙ (L[2],L[3])¿ª µ—¥Ÿ ¡ˆ≥™¥¬ º±
                //lrame_ine(fL, lines[line_index].start, lines[line_index].end, Scalar(255,255,0), 2);
                
                //cur.end= Point((int)L[0], (int)L[1]);
                //cur.start = Point((int)L[2], (int)L[3]);
                
                //end_y∞° ¡¶¿œ ¿€¿∫ ∞™¿ª √£æ∆ «ˆ¿Á ¬˜º±¿ª √£¥¬¥Ÿ.
                //yseq[0]¿Ã «ˆ¿Á øﬁ¬  ¬˜º±
                //yseq.push_back(cur);
                //sort(yseq.begin() , yseq.end(), lessY);
    
                //vector<Point> current;
                //current.push_back(line[0]); //start
                //current.push_back(line[1]); //end
                
            //    line(frame_L, yseq[0].start, yseq[0].end, Scalar(255, 255, 0), 2, 0);

                vector<Point> Lane;
                vector<Point> fillLane;
                Point crv, crx, cry;

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

                fillConvexPoly(curlane, &fillLane[0], fillLane.size(), Scalar(255, 255, 0), 8, 0);

                addWeighted(roiframe, 0.7, curlane, 0.2, 0.0, roiframe);

                circle(frame_L, Point((int)L[0], (int)L[1]), 5, Scalar(0, 255, 0));
                circle(frame_L, Point((int)L[2], (int)L[3]), 5, Scalar(0, 0, 255));
                line_index++;
                
                Vec4i longL = blob_lines[longDistanceIndex];

                imshow("frame", frame);
            }
        }
    }
    lines->size = line_index;
    return line_index;
}

int LaneDetect::extractLine_R(Mat &frame, Mat &roiframe, Mat &frame_R, CLine * lines, int num_labels, Mat edge_img, Mat &img_labels, Mat stats, Mat centroids)
{
    vector<Vec4i> blob_lines;    //º±¿ª ∞®¡ˆ«œ±‚ ¿ß«— ∏∂¡ˆ∏∑ ¡°¿ª ∆˜«‘«— ∫§≈Õ
    int line_index = 0;
    curPoint cur;
    vector<curPoint> yseq;
    
    // Labelled image blob¿ª ∏µŒ √º≈© (i=0¿Œ ∞ÊøÏ¥¬ πË∞Ê¿Ãπ«∑Œ ¡¶ø‹)
    for (int i = 1; i < num_labels; i++)
    {
        // blob area∞° ≥ π´ ¿€¿∏∏È π´Ω√
        if (stats.at<int>(i, CC_STAT_AREA) < MIN_AREA_PIXELS) continue;

        //Houghtrasform »ƒ line ¡∂∞¢ √ﬂ√‚
        HoughLinesP(edge_img, blob_lines, 1, CV_PI / 180, 100, 100, 2);
        //∞À√‚«— º± ±◊∏Æ±‚
        drawlines(frame_R, img_labels);

        //line¿ª «œ≥™µµ √ﬂ√‚ ∏¯«ﬂ¿ª ∞ÊøÏ ¿Ã»ƒ∏¶ π´Ω√«œ∞Ì ¥Ÿ¿Ω blob¿∏∑Œ ≥—æÓ∞®
        if (blob_lines.size() == 0) continue;

        //√ﬂ√‚«— line ¡∂∞¢
        for (int k = 0; k < blob_lines.size(); k++)
        {
            // ¿ÃπÃ¡ˆ blob¿« center∏¶ ±‚¡ÿ¿∏∑Œ ªı∑ŒøÓ line ª˝º∫
            Vec4i L = blob_lines[k];

            double rate = ((double)(L[3] - L[1])) / ((double)(L[2] - L[0]) + 0.5);
            
            if (rate < 0.5 && rate > 1) break; //ø¿∏•¬ øµø™¿« º±¿« ±‚øÔ±‚ ¡∂∞«ø° ∫Œ«’«œ¡ˆ æ ¿∏∏È break

                                                   // ∏µÁ ¬˜º± ¿˙¿Â
            int unique_line_flag = 1;

            // ¿ÃπÃ ª˝º∫«— line∞˙ ¡ﬂ∫πµ«¥¬(≥ π´ ∞°±ÓøÓ) line¿∫ ªı∑Œ ª˝º∫«œ¡ˆ æ ¥¬¥Ÿ
            for (int j = 0; j < line_index; j++)
            {
                if (lines[j].dist_to_point(Point(L[0], L[1])) < LINE_SPACING_PIXELS)
                    unique_line_flag = 0;
            }

            // ªı∑ŒøÓ line¿Œ ∞ÊøÏ ª˝º∫«œø© lines ∏ﬁ∏∏Æ øµø™ø° ¿˙¿Â
            if (unique_line_flag)
            {

                double centerx = centroids.at<double>(i, 0);
                double centery = centroids.at<double>(i, 1);
                
                int left = stats.at<int>(i, CC_STAT_LEFT);
                int width = stats.at<int>(i, CC_STAT_WIDTH);

                lines[line_index] = CLine(rate, Point((int)centerx, (int)centery), left, width);
                
                //∞À√‚«— º±¿∫ ¡° (L[0],L[1])∞˙ (L[2],L[3])¿ª µ—¥Ÿ ¡ˆ≥™¥¬ º±
                    line(frame_R, lines[line_index].start, lines[line_index].end, Scalar(255,255,0), 2);
                
                Mat curlane(roiframe.rows, roiframe.cols, CV_8UC3);
                Mat temp(roiframe.rows, roiframe.cols, CV_8UC1);
                for (int i = 0; i<temp.cols; i++)
                    for (int j = 0; j<temp.rows; j++)
                        temp.at<uchar>(Point(i, j)) = 0;


                cur.end = Point((int)L[0], (int)L[1]);
                cur.start = Point((int)L[2], (int)L[3]);

                //end_y∞° ¡¶¿œ ¿€¿∫ ∞™¿ª √£æ∆ «ˆ¿Á ¬˜º±¿ª √£¥¬¥Ÿ.
                //yseq[0]¿Ã «ˆ¿Á øﬁ¬  ¬˜º±
                yseq.push_back(cur);
                sort(yseq.begin(), yseq.end(), lessY);

                vector<Point> current;
//                current.push_back(line[0]); //start
//                current.push_back(line[1]); //end
                
                line(frame_R, yseq[0].start, yseq[0].end, Scalar(255, 255, 0), 8, 0);
                
                line(temp,yseq[0].start, yseq[0].end, 255, 3);
                addWeighted(roiframe, 0.7, curlane, 0.2, 0.0, roiframe);

                circle(frame_R, Point((int)L[0], (int)L[1]), 5, Scalar(0, 255, 0));
                    circle(frame_R, Point((int)L[2], (int)L[3]), 5, Scalar(0, 0, 255));
                line_index++;
                
                imshow("frame", frame);
            }
        }
    }

    lines->size = line_index;
    return line_index;
}


void LaneDetect::getCurrentlane(Mat& image, double* angle, Point &curv, Point &curx, Point &cury, int seq_ey_R, int seq_ey_L, CLine* lines_R, CLine* lines_L, int right_lines, int left_lines, int *check, int width, int height) {
    double al, bl, ar, br;
    int x, y;
    int num_lines = left_lines;
    double x1L, x2L, y1L, y2L, x1R, x2R, y1R, y2R;
    vector<double> angles;
    vector<Point> X;
    vector<Point> Y;
    vector<Point> V;
    vector<Point> ROI_Vertices;

    int index = 0;
    

    Mat mask = Mat(image.rows, image.cols, CV_8UC1);

    for (int k = 0; k < image.cols; k++)
        for (int m = 0; m < image.rows; m++)
            mask.at<uchar>(Point(k, m)) = 0;

    Mat imageDest = Mat(image.rows, image.cols, CV_8UC3);
    vector<Point> ROI_Poly;
    Point v;
    if (left_lines > right_lines) {
        num_lines = right_lines;

    }
    V.resize(num_lines);
    X.resize(num_lines);
    Y.resize(num_lines);
    angles.resize(num_lines);

    for (int i = 0; i < num_lines; i++) {

        //∞¢ øµø™¿« ¡°¿Œ x,y¿« ±≥¡°¿ª ±∏«‘

        x1L = lines_L[i].left;
        x2L = lines_L[i].left + lines_L[i].width;
        y1L = lines_L[i].rate * x1L + lines_L[i].y_inter;
        y2L = lines_L[i].rate * x2L + lines_L[i].y_inter;
        //circle(image, Point(x1L,y1L+height), 5, Scalar(255, 0, 0), 5);
        //circle(image, Point(x2L, y2L+height), 5, Scalar(255, 0, 0), 5);

        x1R = lines_R[i].left;
        x2R = lines_R[i].left + lines_R[i].width;
        y1R = lines_R[i].rate * x1R + lines_R[i].y_inter;
        y2R = lines_R[i].rate * x2R + lines_R[i].y_inter;

        //ø¿∏•¬  øµø™¿Ãπ«∑Œ ¥ı«‘
        x1R += (image.cols / 2);
        x2R += (image.cols / 2);

        al = (y2L - y1L) / (x2L - x1L);
        bl = y1L - (y2L - y1L) / (x2L - x1L)*x1L;
        ar = (y2R - y1R) / (x2R - x1R);
        br = y1R - (y2R - y1R) / (x2R - x1R)*x1R;

        x = (int)(-(bl - br) / (al - ar));
        y = (int)(al*(-(bl - br) / (al - ar)) + bl);

        //»≠∏È «•Ω√∏¶ ¿ß«ÿ

        //circle(image, Point(x1R, y1R+height), 5, Scalar(255,0 , 0), 5);
        //circle(image, Point(x2R, y2R+height), 5, Scalar(255,0, 0), 5);
        //circle(image, Point(x, y+height), 5, Scaqlar(0, 0, 255), 5);
        if ((x >= 0 && x <= mask.cols) && (y >= 0 && y <= mask.rows))
        {
            //º“Ω«¡° ¿˙¿Â
            V[i] = Point(x, y);
            //circle(image, V[i], 5, Scalar(0, 0, 255), 3);
            X[i] = Point(lines_L[i].start.x, lines_L[i].start.y);
            Y[i] = Point(lines_R[i].end.x + image.cols / 2, lines_R[i].end.y);

            //∞¢∞¢ øﬁ¬ ,ø¿∏•¬ øµø™æ»ø° ¿÷¿∏∏È ¡¯«‡
            if ((X[i].x >= 0 && X[i].x <= image.cols / 2) && (Y[i].x > image.cols / 2 && Y[i].x <= image.cols))
            {

                if (X[i].y > Y[i].y) {
                    double y_inter = lines_L[i].y_inter;
                    double rate = lines_L[i].rate;
                    X[i].y = Y[i].y;
                    X[i].x = (X[i].y - y_inter) / rate;    //y = rate * x + y_inter;y_inter = -rate*center.x + center.y;
                }
                else {
                    double rate = lines_R[i].rate;
                    double y_inter = lines_R[i].y_inter;
                    double d = Y[i].y - X[i].y;
                    Y[i].y = X[i].y;
                    Y[i].x = Y[i].y *(1 / rate) + image.cols / 2;

                }

                //circle(image, X[i], 5, Scalar(0, 255, 0), 3);
                //circle(image, Y[i], 5, Scalar(0, 255, 0), 3);
                
                
                
                angles[i] = getAngle(X[i], V[i], Y[i]);

                if (*angle > angles[i])
                {
                    *angle = angles[i];
                    index = i;

                    curv = V[index];
                    curx = X[index];
                    cury = Y[index];

                    (*check)++;
                }
                
                ROI_Vertices.push_back(curv);
                ROI_Vertices.push_back(curx);
                ROI_Vertices.push_back(cury);

                approxPolyDP(ROI_Vertices, ROI_Poly, 1.0, true);
                fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);


                image.copyTo(imageDest, mask);
            }

        }

    }

}
    /*
void LaneDetect::currentLine_L(Mat & img, , CLine * lines, CLine * curlines, int num_labels, Mat stats, Mat centroids)
{
    int cur_index = 0;
    double largerate = 0;
    char text[15];

    for (int i = 1; i < num_labels; i++)
    {

        for (int k = 0; k < Linfo->blob_lines.size(); k++)
        {

            Vec4i L = Linfo->blob_lines[k];

            double rate = ((double)(L[3] - L[1])) / ((double)(L[2] - L[0]) + 0.5);

            // «ˆ¿Á¬˜º± ¿˙¿Â
            if (abs(rate) > abs(largerate)) {    //±◊¡ﬂ ±‚øÔ±‚¿« ¿˝¥Ò∞™¿Ã ∞°¿Â≈´ º±¿Ã «ˆ¿Á º±
                largerate = rate;

                //¡ﬂ∫π
                double centerx = centroids.at<double>(i, 0);
                double centery = centroids.at<double>(i, 1);
                int left = stats.at<int>(i, CC_STAT_LEFT);
                int width = stats.at<int>(i, CC_STAT_WIDTH);

                CLine current = CLine(largerate, Point((int)centerx, (int)centery), left, width);

                // curlines ∏ﬁ∏∏Æ øµø™ø° ¿˙¿Â
                curlines[cur_index] = current;
                cur_index++;
                //circle(img, Point(current.start.x, current.start.y + img.rows / 5 * 3), 5, Scalar(255, 255, 0), 5);
                sprintf_s(text, "c%.2lfpi", current.rate);
                putText(img, text, Point(lines[i].center.x,lines[i].center.y+img.rows/5*3), 2, 0.5, Scalar(0, 0, 0));
                //circle(img, Point(current.end.x, current.end.y + img.rows / 5 * 3), 5, Scalar(255, 255, 0), 5);
            }
        }
    }
}
*/
/*
void LaneDetect::currentLane(Mat& image, double* angle, Point &curv, Point &curx, Point &cury, CLine* lines_R, CLine* lines_L, int right_lines, int left_lines, int *check,int width, int height) {
    double al, bl, ar, br;
    int x, y;
    int num_lines = left_lines;
    double x1L, x2L, y1L, y2L, x1R, x2R, y1R, y2R;
    vector<double> angles;
    vector<Point> X;
    vector<Point> Y;
    vector<Point> V;
    vector<Point> ROI_Vertices;

    int index = 0;


    Mat mask = Mat(image.rows, image.cols, CV_8UC1);

    for (int k = 0; k < image.cols; k++)
        for (int m = 0; m < image.rows; m++)
            mask.at<uchar>(Point(k, m)) = 0;

    Mat imageDest = Mat(image.rows, image.cols, CV_8UC3);
    vector<Point> ROI_Poly;
    Point v;
    if (left_lines > right_lines) {
        num_lines = right_lines;

    }
    V.resize(num_lines);
    X.resize(num_lines);
    Y.resize(num_lines);
    angles.resize(num_lines);

    for (int i = 0; i < num_lines; i++) {

        //∞¢ øµø™¿« ¡°¿Œ x,y¿« ±≥¡°¿ª ±∏«‘
        
        x1L = lines_L[i].left;
        x2L = lines_L[i].left + lines_L[i].width;
        y1L = lines_L[i].rate * x1L + lines_L[i].y_inter;
        y2L = lines_L[i].rate * x2L + lines_L[i].y_inter;
        //circle(image, Point(x1L,y1L+height), 5, Scalar(255, 0, 0), 5);
        //circle(image, Point(x2L, y2L+height), 5, Scalar(255, 0, 0), 5);
        
        x1R = lines_R[i].left;
        x2R = lines_R[i].left + lines_R[i].width;
        y1R = lines_R[i].rate * x1R + lines_R[i].y_inter;
        y2R = lines_R[i].rate * x2R + lines_R[i].y_inter;

        //ø¿∏•¬  øµø™¿Ãπ«∑Œ ¥ı«‘
        x1R += (image.cols / 2);
        x2R += (image.cols / 2);

        al = (y2L - y1L) / (x2L - x1L);
        bl = y1L - (y2L - y1L) / (x2L - x1L)*x1L;
        ar = (y2R - y1R) / (x2R - x1R);
        br = y1R - (y2R - y1R) / (x2R - x1R)*x1R;

        x = (int)(-(bl - br) / (al - ar));
        y = (int)(al*(-(bl - br) / (al - ar)) + bl);

        //»≠∏È «•Ω√∏¶ ¿ß«ÿ
        
        //circle(image, Point(x1R, y1R+height), 5, Scalar(255,0 , 0), 5);
        //circle(image, Point(x2R, y2R+height), 5, Scalar(255,0, 0), 5);
        //circle(image, Point(x, y+height), 5, Scaqlar(0, 0, 255), 5);
        if ((x >= 0 && x <= mask.cols) && (y >= 0 && y <= mask.rows))
        {
            //º“Ω«¡° ¿˙¿Â
            V[i] = Point(x, y);
            //circle(image, V[i], 5, Scalar(0, 0, 255), 3);
            X[i] = Point(lines_L[i].start.x, lines_L[i].start.y);
            Y[i] = Point(lines_R[i].end.x + image.cols / 2, lines_R[i].end.y);

            //∞¢∞¢ øﬁ¬ ,ø¿∏•¬ øµø™æ»ø° ¿÷¿∏∏È ¡¯«‡
            if ((X[i].x >= 0 && X[i].x <= image.cols / 2) && (Y[i].x > image.cols / 2 && Y[i].x <= image.cols))
            {

                if (X[i].y > Y[i].y) {
                    double y_inter = lines_L[i].y_inter;
                    double rate = lines_L[i].rate;
                    X[i].y = Y[i].y;
                    X[i].x = (X[i].y - y_inter) / rate;    //y = rate * x + y_inter;y_inter = -rate*center.x + center.y;
                }
                else {
                    double rate = lines_R[i].rate;
                    double y_inter = lines_R[i].y_inter;
                    double d = Y[i].y - X[i].y;
                    Y[i].y = X[i].y;
                    Y[i].x = Y[i].y *(1 / rate) + image.cols / 2;
                    
                }

                //circle(image, X[i], 5, Scalar(0, 255, 0), 3);
                //circle(image, Y[i], 5, Scalar(0, 255, 0), 3);
                angles[i] = getAngle(X[i], V[i], Y[i]);

                if (*angle > angles[i])
                {
                    *angle = angles[i];
                    index = i;

                    curv = V[index];
                    curx = X[index];
                    cury = Y[index];

                    (*check)++;
                }

                ROI_Vertices.push_back(curv);
                ROI_Vertices.push_back(curx);
                ROI_Vertices.push_back(cury);

                approxPolyDP(ROI_Vertices, ROI_Poly, 1.0, true);
                fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);


                image.copyTo(imageDest, mask);
            }

        }

    }


}
*/

void LaneDetect::displayLineinfo(Mat img, CLine * lines, int num_lines, Scalar linecolor, Scalar captioncolor, int width, int height) {
    for (int i = 0; i < num_lines; i++) {
        double x1, x2, y1, y2;

        /*img∞° ¿¸√º frame¿Ã±‚ ∂ßπÆø° width, height¿ª ¥ı«ÿ¡‹
        roiframe¿Ã∂Û∏È ¥ı«ÿ¡Ÿ « ø‰ æ¯¿Ω*/
        x1 = lines[i].left;
        x2 = lines[i].left + lines[i].width;
        y1 = lines[i].rate * x1 + lines[i].y_inter;
        y2 = lines[i].rate * x2 + lines[i].y_inter;
        line(img, Point((int)x1 + width, (int)y1 + height), Point((int)x2 + width, (int)y2 + height), linecolor, 1, LINE_AA);

        char text[15];
        double tanval = -atan2(lines[i].rate, 1.0) / (double)CV_PI;

        if (tanval < 0)
            tanval = tanval + 1.0;
//        sprintf_s(text, "%.2lfpi", tanval);
        putText(img, text, lines[i].center, 2, 0.5, captioncolor);

    }
}
void LaneDetect::detectcolor(Mat& image, double minH, double maxH, double minS, double maxS, Mat& mask) {
    Mat hsv;
    vector<Mat> channels;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    split(hsv, channels);
    Mat mask1;
    threshold(channels[0], mask1, maxH, 255, THRESH_BINARY_INV);
    Mat mask2;
    threshold(channels[0], mask2, minH, 255, THRESH_BINARY);
    Mat hmask;
    if (minH < maxH)
        hmask = mask1 & mask2;
    else
        hmask = mask1 | mask2;

    threshold(channels[1], mask1, maxS, 255, THRESH_BINARY_INV);
    threshold(channels[1], mask2, minS, 255, THRESH_BINARY);

    Mat smask;
    smask = mask1 & mask2;
    mask = hmask&smask;

}
double LaneDetect::getAngle(Point a, Point b, Point c) {
    Point ab = { b.x - a.x,b.y - a.y };
    Point cb = { b.x - c.x,b.y - c.y };

    double inner = (ab.x*cb.x + ab.y*cb.y);
    double l1 = sqrt(ab.x*ab.x + ab.y*ab.y);
    double l2 = sqrt(cb.x*cb.x + cb.y*cb.y);

    double lx1 = ab.x / l1;
    double ly1 = ab.y / l1;
    double lx2 = cb.x / l2;
    double ly2 = cb.y / l2;

    inner = (lx1*lx2 + ly1*ly2);
    double result = acos(inner) * 180 / CV_PI;

    return result;
}
