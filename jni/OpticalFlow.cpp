#include <jni.h>
#include <com_example_opticalflow_OpticalFlow.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <converters.h>
#include <android/log.h>

#define LOG_TAG "OpticalFlow"

#define LOGI(fmt, args...) __android_log_write(ANDROID_LOG_INFO, LOG_TAG, fmt, ##args)
#define LOGD(fmt, args...) __android_log_write(ANDROID_LOG_DEBUG, LOG_TAG, fmt, ##args)
#define LOGE(fmt, args...) __android_log_write(ANDROID_LOG_ERROR, LOG_TAG, fmt, ##args)

#define LOGG(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)

using namespace cv;
using namespace std;

JNIEXPORT void JNICALL Java_com_example_opticalflow_OpticalFlow_opticalFlow
(JNIEnv *env, jclass clazz, jlong inputImage, jint maxCorners, jdouble qualityLevel, jdouble minDistance){
	Mat* inputImage1 = (Mat*) inputImage;
	Mat mRgba = inputImage1->clone();
    static Mat matOpFlowThis,matOpFlowPrev;
    static vector<Point2f> mMOPcorners;
    static vector<Point2f> cornersPrev,cornersThis;
    static vector<Point2f> mMOP2fptsPrev, mMOP2fptsSafe, mMOP2fptsThis;
    static vector<u_char> mMOBStatus;
    static vector<u_char> byteStatus;
    static vector<float> mMOFerr;
    static Mat vectorPoint2Mat;
    static Mat vectorPoint2f2Mat;

    if(mMOP2fptsPrev.empty() ==true){
    	cvtColor(mRgba,matOpFlowThis,CV_RGBA2GRAY);
        matOpFlowThis.copyTo(matOpFlowPrev);
    	goodFeaturesToTrack(matOpFlowPrev, mMOPcorners, maxCorners,qualityLevel,minDistance);
//    	vector_Point_to_Mat(mMOPcorners,vectorPoint2Mat);
//    	Mat_to_vector_Point2f(vectorPoint2Mat,mMOP2fptsPrev);
    	mMOP2fptsPrev = mMOPcorners;
    	mMOP2fptsSafe = mMOP2fptsPrev;
    }
    else{
    	 matOpFlowThis.copyTo(matOpFlowPrev);
    	 cvtColor(mRgba,matOpFlowThis,CV_RGBA2GRAY);
    	 goodFeaturesToTrack(matOpFlowPrev, mMOPcorners, maxCorners,qualityLevel,minDistance);
//    	 vector_Point_to_Mat(mMOPcorners,vectorPoint2Mat);
//    	 Mat_to_vector_Point2f(vectorPoint2Mat,mMOP2fptsThis);
    	 mMOP2fptsThis = mMOPcorners;
    	 mMOP2fptsPrev = mMOP2fptsSafe;
    	 mMOP2fptsSafe = mMOP2fptsThis;
    }
    calcOpticalFlowPyrLK(matOpFlowPrev, matOpFlowThis, mMOP2fptsPrev, mMOP2fptsThis, mMOBStatus, mMOFerr);
//    vector_Point2f_to_Mat(mMOP2fptsPrev,vectorPoint2f2Mat);
//    Mat_to_vector_Point(vectorPoint2Mat,cornersPrev);
//    vector_Point2f_to_Mat(mMOP2fptsThis,vectorPoint2f2Mat);
//    Mat_to_vector_Point(vectorPoint2Mat,cornersThis);
    cornersPrev = mMOP2fptsPrev;
    cornersThis = mMOP2fptsThis;
    byteStatus = mMOBStatus;
    int y = byteStatus.size()-1;
    LOGG("Size:%d",cornersPrev.size());
    for (int x = 0; x < y; x++) {
                if (byteStatus.at(x) == 1) {
                	CvPoint p1, p2;
                    p1 = cornersThis.at(x);
                    p2 = cornersPrev.at(x);

                    circle(mRgba, p1, 5, Scalar(255, 0, 0), 3 - 1);

                    line(mRgba, p1, p2, Scalar(255, 0, 0), 3);
                    }
                }
}
