LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#
LOCAL_C_INCLUDES := $(LOCAL_PATH)
OPENCV_LIB_TYPE:=STATIC
OPENCV_INSTALL_MODULES:=on
OPENCV_CAMERA_MODULES:=off
include D:/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk
#
#

LOCAL_MODULE    := OpticalFlow
LOCAL_SRC_FILES := OpticalFlow.cpp

include $(BUILD_SHARED_LIBRARY)
