package com.example.opticalflow;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.utils.Converters;
import org.opencv.video.Video;

import android.os.Bundle;
import android.provider.ContactsContract.CommonDataKinds.Im;
import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class MainActivity extends Activity implements CvCameraViewListener2 {

	private static final String TAG = "Optical::Activity";
	private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
	public static final int JAVA_DETECTOR = 0;

	private MenuItem mItemFace50;
	private MenuItem mItemFace40;
	private MenuItem mItemFace30;
	private MenuItem mItemFace20;
	private MenuItem mItemFace10;

	private Mat mRgba;
	private Mat mGray;
	private Mat flip;
	private Rect roi;
	private File mCascadeFile;
	private CascadeClassifier mJavaDetector;

	private int mDetectorType = JAVA_DETECTOR;
	private String[] mDetectorName;

	private float mRelativeFaceSize = 0.2f;
	private int mAbsoluteFaceSize = 0;

	private CameraBridgeViewBase mOpenCvCameraView;

	private Mat matOpFlowThis, matOpFlowPrev, hsvImage, bgrImage, maskImage;
	private MatOfPoint MOPcorners;
	private MatOfPoint2f mMOP2fptsPrev, mMOP2fptsSafe, mMOP2fptsThis;
	private List<org.opencv.core.Point> cornersPrev;
	private List<org.opencv.core.Point> cornersThis;
	private MatOfByte mMOBStatus;
	private List<Byte> byteStatus;
	private int i, j;
	private org.opencv.core.Point pt, pt2;
	private Scalar colorRed = new Scalar(255, 0, 0);
	private MatOfFloat mMOFerr;
	private int maxCorners = 200;
	private Mat cropImage;
	private boolean croped = true;

	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	boolean useHarrisDetector = false;
	boolean isRgb = false;
	double k = 0.04;
	Size size = new Size();
	Rect thisRect, preRect;

	private enum DIRECTION {
		LEFT, RIGHT, UP, DOWN, NONE
	};

	DIRECTION direction;

	OpticalFlow opticalFlow;
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");

				try {
					// load cascade file from application resources
					InputStream is = getResources().openRawResource(
							R.raw.cascade);
					File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
					mCascadeFile = new File(cascadeDir, "cascade.xml");
					FileOutputStream os = new FileOutputStream(mCascadeFile);

					byte[] buffer = new byte[4096];
					int bytesRead;
					while ((bytesRead = is.read(buffer)) != -1) {
						os.write(buffer, 0, bytesRead);
					}
					is.close();
					os.close();

					mJavaDetector = new CascadeClassifier(
							mCascadeFile.getAbsolutePath());
					if (mJavaDetector.empty()) {
						Log.e(TAG, "Failed to load cascade classifier");
						mJavaDetector = null;
					} else
						Log.i(TAG, "Loaded cascade classifier from "
								+ mCascadeFile.getAbsolutePath());

					cascadeDir.delete();

				} catch (IOException e) {
					e.printStackTrace();
					Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
				}

				mOpenCvCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.face_detect_surface_view);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
		// // Select Camera
		mOpenCvCameraView.setCameraIndex(1);
		mOpenCvCameraView.setMaxFrameSize(720, 540);
		size.height = 540;
		size.width = 720;
		// Creat preRect 50x50
		preRect = new Rect((int) size.width / 2 - 25,
				(int) size.height / 2 - 25, 50, 50);
		thisRect = new Rect();
		mOpenCvCameraView.setCvCameraViewListener(this);
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
				mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mGray = new Mat();
		mRgba = new Mat();
		mMOP2fptsPrev = new MatOfPoint2f();
		mMOP2fptsSafe = new MatOfPoint2f();
		mMOP2fptsThis = new MatOfPoint2f();
		matOpFlowThis = new Mat();
		matOpFlowThis = new Mat();
		matOpFlowPrev = new Mat();
		MOPcorners = new MatOfPoint();
		mMOBStatus = new MatOfByte();
		mMOFerr = new MatOfFloat();
		hsvImage = new Mat();
		bgrImage = new Mat();
		maskImage = new Mat();
		cropImage = new Mat();
		flip = new Mat();
		pt = new Point();
		pt2 = new Point();

	}

	public void onCameraViewStopped() {
		mGray.release();
		mRgba.release();
		flip.release();
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

		// Log.e("Test", "Test");
		flip = inputFrame.rgba();
		Core.flip(flip, mRgba, 1);
		// Get Size Image
		size.height = flip.height();
		size.width = flip.width();

		// mRgba = inputFrame.rgba();
		Imgproc.cvtColor(mRgba, bgrImage, Imgproc.COLOR_RGBA2BGR);
		Imgproc.cvtColor(bgrImage, hsvImage, Imgproc.COLOR_BGR2HSV);
		mGray = inputFrame.gray();
		Core.inRange(hsvImage, new Scalar(0, 30, 76), new Scalar(14, 128, 180),
				maskImage);
		// Imgproc.cvtColor(maskImage, mRgba,Imgproc.COLOR_GRAY2RGBA);
		if (isRgb) {
			if (mAbsoluteFaceSize == 0) {
				int height = mGray.rows();
				if (Math.round(height * mRelativeFaceSize) > 0) {
					mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
				}

			}

			MatOfRect faces = new MatOfRect();

			if (mJavaDetector != null)
				mJavaDetector.detectMultiScale(mGray, faces, 1.1,
						2,
						2, // TODO:
							// objdetect.CV_HAAR_SCALE_IMAGE
						new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
						new Size());
			else {
				Log.e(TAG, "Detection method is not selected!");
			}

			Rect[] facesArray = faces.toArray();
			for (int i = 0; i < facesArray.length; i++) {
				// Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(),
				// FACE_RECT_COLOR, 3);
				roi = new Rect((int) facesArray[0].tl().x,
						(int) (facesArray[0].tl().y), facesArray[0].width,
						(int) (facesArray[0].height));
				cropImage = mRgba.submat(roi);
				croped = true;
				// Log.e("Croped", roi.x + "|" + roi.y);
				break;
			}
			//
			if (croped) {
				if (mMOP2fptsPrev.rows() == 0) {

					// get this mat
					Imgproc.cvtColor(mRgba, matOpFlowThis,
							Imgproc.COLOR_RGBA2GRAY);

					// copy that to prev mat
					matOpFlowThis.copyTo(matOpFlowPrev);

					// get prev corners
					// Imgproc.goodFeaturesToTrack(matOpFlowPrev, MOPcorners,
					// maxCorners, qualityLevel, minDistance);
					Imgproc.goodFeaturesToTrack(matOpFlowPrev, MOPcorners,
							maxCorners, qualityLevel, minDistance, maskImage,
							blockSize, useHarrisDetector, k);
					mMOP2fptsPrev.fromArray(MOPcorners.toArray());

					// get safe copy of this corners
					mMOP2fptsPrev.copyTo(mMOP2fptsSafe);
				} else {
					// we've been through before so
					// this mat is valid. Copy it to prev mat
					matOpFlowThis.copyTo(matOpFlowPrev);

					// get this mat
					Imgproc.cvtColor(mRgba, matOpFlowThis,
							Imgproc.COLOR_RGBA2GRAY);

					// get the corners for this mat
					// Imgproc.goodFeaturesToTrack(matOpFlowThis, MOPcorners,
					// maxCorners, qualityLevel, minDistance);

					Imgproc.goodFeaturesToTrack(matOpFlowPrev, MOPcorners,
							maxCorners, qualityLevel, minDistance, maskImage,
							blockSize, useHarrisDetector, k);

					mMOP2fptsThis.fromArray(MOPcorners.toArray());

					// retrieve the corners from the prev mat
					// (saves calculating them again)
					mMOP2fptsSafe.copyTo(mMOP2fptsPrev);

					// and save this corners for next time through

					mMOP2fptsThis.copyTo(mMOP2fptsSafe);
				}

				// matOpFlowPrev = new Mat(mRgba.size(),CvType.CV_8UC1);
				// matOpFlowThis = new Mat(mRgba.size(),CvType.CV_8UC1);
				Video.calcOpticalFlowPyrLK(matOpFlowPrev, matOpFlowThis,
						mMOP2fptsPrev, mMOP2fptsThis, mMOBStatus, mMOFerr);

				cornersPrev = mMOP2fptsPrev.toList();
				cornersThis = mMOP2fptsThis.toList();
				byteStatus = mMOBStatus.toList();

				j = byteStatus.size() - 1;
				ArrayList<Double> listAngle2 = new ArrayList<Double>();
				ArrayList<Double> listHypotenuse = new ArrayList<Double>();
				for (i = 0; i < j; i++) {
					if (byteStatus.get(i) == 1) {
						// for(int i=0;i< mMOP2fptsThis.total();i++){
						// Log.e("Test",roi.x+"|"+roi.y);
						// pt.x = roi.x + cornersThis.get(i).x;
						// pt.y = roi.y + cornersThis.get(i).y;
						// pt2.x = roi.x + cornersPrev.get(i).x;
						// pt2.y = roi.y + cornersPrev.get(i).y;
						pt = cornersThis.get(i);
						pt2 = cornersPrev.get(i);
						if (Math.abs(pt2.y - pt.y) >= 10
								|| Math.abs(pt2.x - pt.x) >= 10) {
							listAngle2.add(Math.atan2(pt2.y - pt.y, pt2.x
									- pt.x));
							listHypotenuse.add(Math.sqrt(Math.pow(pt.x - pt2.x,
									2) + Math.pow(pt.y - pt2.y, 2)));
						}

						Core.circle(mRgba, pt, 5, colorRed, 3 - 1);

						Core.line(mRgba, pt, pt2, colorRed, 3);
					}
				}
				thisRect = detect(listAngle2, listHypotenuse, preRect);
				Core.rectangle(mRgba, thisRect.tl(), thisRect.br(),
						FACE_RECT_COLOR, 3);
				preRect = thisRect;
			}

			return mRgba;
		} else

			return maskImage;
	}

	public Rect detect(ArrayList<Double> listAngle2,
			ArrayList<Double> listHypotenuse, Rect preRect) {
		Point point11 = new Point(0, 0);
		Point point12 = new Point(10, 0);
		// List Angle
		ArrayList<Double> listAngleRight = new ArrayList<Double>();
		ArrayList<Double> listAngleLeft = new ArrayList<Double>();
		ArrayList<Double> listAngleUp = new ArrayList<Double>();
		ArrayList<Double> listAngleDown = new ArrayList<Double>();
		// List Length
		ArrayList<Double> listHypotenuseRight = new ArrayList<Double>();
		ArrayList<Double> listHypotenuseLeft = new ArrayList<Double>();
		ArrayList<Double> listHypotenuseUp = new ArrayList<Double>();
		ArrayList<Double> listHypotenuseDown = new ArrayList<Double>();
		double angle1 = Math
				.atan2(point11.y - point12.y, point11.x - point12.x);
		for (int i = 0; i < listAngle2.size(); i++) {
			double angle = (angle1 - listAngle2.get(i)) * 360 / (2 * 3.14);
			// Right
			if ((angle >= 0 && angle < 45) || (angle >= 315 && angle <= 360)) {
				listAngleRight.add(angle);
				listHypotenuseRight.add(listHypotenuse.get(i));
				// Log.e(TAG, angle+"");
			}
			// Left
			if (angle >= 135 && angle < 225) {
				listAngleLeft.add(angle);
				listHypotenuseLeft.add(listHypotenuse.get(i));
				// Log.e(TAG, angle+"");
			}
			// Up
			if (angle >= 45 && angle < 135) {
				listAngleUp.add(angle);
				listHypotenuseUp.add(listHypotenuse.get(i));
				// Log.e(TAG, angle+"");
			}
			// Down
			if (angle >= 225 && angle < 315) {
				// Log.e(TAG, angle+"");
				listAngleDown.add(angle);
				listHypotenuseDown.add(listHypotenuse.get(i));
			}
		}
		int index = checkListMax(listAngleLeft, listAngleRight, listAngleUp,
				listAngleDown);
		Rect thisRect = preRect;
		switch (index) {
		case 0:
			direction = DIRECTION.LEFT;
			thisRect.x = preRect.x
					- getAverage(listHypotenuseLeft, listAngleLeft);
			if (thisRect.x < 0)
				thisRect.x = 0;
			Log.e(TAG, "LEFT");
			break;
		case 1:
			direction = DIRECTION.RIGHT;
			thisRect.x = preRect.x
					+ getAverage(listHypotenuseRight, listAngleRight);
			if (thisRect.x > size.width)
				thisRect.x = (int) size.width;
			Log.e(TAG, "RIGHT");
			break;
		case 2:
			direction = DIRECTION.UP;
			thisRect.y = preRect.y - getAverage(listHypotenuseUp, listAngleUp);
			if (thisRect.y < 0)
				thisRect.y = 0;
			Log.e(TAG, "UP");
			break;
		case 3:
			direction = DIRECTION.DOWN;
			thisRect.y = preRect.y
					+ getAverage(listHypotenuseDown, listAngleDown);
			if (thisRect.y > size.height)
				thisRect.y = (int) size.height;
			Log.e(TAG, "DOWN");
			break;
		case 4:
			direction = DIRECTION.NONE;
			// Log.e(TAG, "NONE");
			break;
		default:
			break;
		}

		return thisRect;
	}

	public int getAverage(ArrayList<Double> listHypotenuse,
			ArrayList<Double> listAngle) {
		ArrayList<Double> list = new ArrayList<Double>();
		for (int i = 0; i < listHypotenuse.size(); i++) {
			list.add(Math.abs(listHypotenuse.get(i)
					* Math.sin(listAngle.get(i))));
		}
		int average = 0;
		double sum = 0;
		for (int j = 0; j < list.size(); j++)
			sum = sum + list.get(j);
		average = (int) sum / list.size();
		return average;
	}

	public int checkListMax(ArrayList<Double> listLeft,
			ArrayList<Double> listRight, ArrayList<Double> listUp,
			ArrayList<Double> listDown) {
		int index = 0;
		int[] listSize = { listLeft.size(), listRight.size(), listUp.size(),
				listDown.size() };
		double max = listSize[0];
		for (int i = 1; i < listSize.length; i++)
			if (listSize[i] > max)
				max = listSize[i];

		for (int j = 0; j < listSize.length; j++)
			if (listSize[j] == max)
				index = j;
		if (max < 25)
			index = 4;
		return index;
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		Log.i(TAG, "called onCreateOptionsMenu");
		mItemFace50 = menu.add("Face size 50%");
		mItemFace40 = menu.add("Face size 40%");
		mItemFace30 = menu.add("Face size 30%");
		mItemFace20 = menu.add("Face size 20%");
		mItemFace10 = menu.add("Face size 10%");
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
		if (item == mItemFace50)
			setMinFaceSize(0.5f);
		else if (item == mItemFace40)
			setMinFaceSize(0.4f);
		else if (item == mItemFace30)
			setMinFaceSize(0.3f);
		else if (item == mItemFace20)
			// setMinFaceSize(0.2f);
			isRgb = false;
		else if (item == mItemFace10) {
			// setMinFaceSize(0.1f);
			isRgb = true;
		}
		return true;
	}

	private void setMinFaceSize(float faceSize) {
		mRelativeFaceSize = faceSize;
		mAbsoluteFaceSize = 0;
	}

}
