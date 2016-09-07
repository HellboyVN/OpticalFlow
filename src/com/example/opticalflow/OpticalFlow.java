package com.example.opticalflow;

public class OpticalFlow {
	public static native void opticalFlow(long inputImage,int maxCorners,double qualityLevel,double minDistance);
	static{
    	System.loadLibrary("OpticalFlow");
    }

}
