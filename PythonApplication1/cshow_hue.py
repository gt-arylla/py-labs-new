//
//  JungleBook.hpp
//  Library
//
//  Created by Arylla CTO on 2018-03-08.
//  Copyright 2018 Arylla. All rights reserved.
//  Used to store functions directly relevant to the Remi Martin project

#include "JungleBook.hpp"
#include "colorspace.hpp"
#include "contour.hpp"
#include "draw.hpp"
#include "roi.hpp"

#include "tester.hpp"

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

namespace jb {
	void black_on_white(Mat imageMat, Mat &imageOutput, Mat &maskOutput, Point2f &originOutput, float &scaleOutput, vector<Mat> &export_pics, string &data_output, int &exception_output, vector<int> test_type) {
		bool shower_switch = 0;
		
		//Keep largest black contour after HSV-V thresholding
		Mat img = imageMat.clone();

		Mat img_ext = color::makeColorspace_single(img, 12);
		
		if (shower_switch) tst::show(img_ext);
		//Otsu threshold
		Mat img_waste;
		double o_thresh=threshold(img_ext, img_waste, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		
		if (o_thresh < 125) threshold(img_ext, img_ext, o_thresh, 255, CV_THRESH_BINARY);
		else threshold(img_ext, img_ext, 75, 255, CV_THRESH_BINARY);
		cout << "Otsu thresh: " << o_thresh<<endl;

		//Find contours
		std::vector<std::vector<cv::Point> > contours;
		std::vector<Vec4i> hierarchy;

		cv::findContours(img_ext, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		//Delete white contours
		vector<vector<cv::Point>> black_contours = contour::white_contour_extraction(contours, hierarchy, img_ext, false);

		//Keep largest contour
		double max_area = 0;
		vector<cv::Point> winner_cnt;
		for (vector<cv::Point> cnt : black_contours) {
			double cnt_area = contourArea(cnt);
			if (cnt_area > max_area) {
				max_area = cnt_area;
				winner_cnt = cnt;
			}
		}

		Mat shower_img = contour::draw_cnt(winner_cnt, img_ext);
		Mat shower_img2 = contour::draw_cnt(contours, img_ext);
		if (shower_switch) tst::show(img);
		if (shower_switch) tst::show(shower_img);
		if (shower_switch) tst::show(shower_img2);

		Rect bndRect = boundingRect(winner_cnt);

		img = img(bndRect);
		shower_img = img_ext(bndRect);
		cv::Point origin = cv::Point(int(img.cols / 2), int(img.rows / 2));
		float scale = sqrt(max_area);

		imageOutput = img;
		maskOutput = shower_img;
		originOutput = origin;
		scaleOutput = scale;
		data_output = "";

		return;
	}

	vector<Mat> junglebook(Mat input_image, vector<Mat> masks, float ROI_x, float ROI_y, float ROI_size, string &user_comments, float &read_result_reference, string mask_path, bool demo, vector<int> test_type) {
		//Admin stuff
		vector<Mat> export_pics;
		bool line_by_line_check = 0;
		bool shower_switch = 0;
		vector<int> demo_test_type = { -1,1 };
        if (demo) {
            test_type = demo_test_type;
        }

		//Make copy of input image
		Mat img = input_image.clone();
		export_pics.push_back(input_image);

		//Apply feature finding
		Mat imgOut, maskOut;
		Point2f originOut;
		float scaleOut;
		string data_output;
		int exception_output = 0;
		if (find(test_type.begin(), test_type.end(), -1) != test_type.end()) {
			black_on_white(img, imgOut, maskOut, originOut, scaleOut, export_pics, data_output, exception_output, test_type);
		
		}
		else {
			if (line_by_line_check) cout << "Feature Start" << endl;
			feature_finder(img, masks, imgOut, maskOut, originOut, scaleOut, export_pics, data_output, exception_output, mask_path, test_type);
			if (line_by_line_check) cout << "Feature End" << endl;
			//Throw error if exception_output is raised
			if (exception_output < 0) {
				read_result_reference = float(exception_output);
				return export_pics;
			}
		}

		//Define masks for image show
	

		Mat circle_mask;
		Mat1b centaur_mask;
		if (find(test_type.begin(), test_type.end(), -1) != test_type.end()) {
			circle_mask = maskOut;
			//centaur_mask = maskOut;
			bitwise_not(maskOut, centaur_mask);
		}
		else {
			centaur_mask = maskOut.clone();
			Mat1b imgOut_ext = color::makeColorspace_single(imgOut, 12); //Index 12 is HSV-V
			double otsu_threshold = color::otsu_8u_with_mask(imgOut_ext, centaur_mask);
			Mat1b centaur_otsu_mask; threshold(imgOut_ext, centaur_otsu_mask, otsu_threshold, 255, THRESH_BINARY);
			bitwise_and(centaur_mask, centaur_otsu_mask, centaur_mask);

			circle_mask = draw::circle_mask(maskOut, originOut, maskOut.cols / 2);
			bitwise_xor(circle_mask, centaur_mask, circle_mask);
		}
		
		if (find(test_type.begin(), test_type.end(), 1) != test_type.end()) {//Quadrant ROI Exports
			//export_pics = {};


		//Split image into quadrants
			vector<Mat> img_quads = draw::quadrant_maker(imgOut);
			vector<Mat> centaur_quads = draw::quadrant_maker(centaur_mask);
			vector<Mat> circle_quads = draw::quadrant_maker(circle_mask);

			//Export images in a number of colorspaces		
			vector<vector<Mat>> cp_Mat_export;
			vector<int> export_cps = { 18,17,16 };  //9 -> YCrCb-Cb, 18 -> Lab-a,18 -> Lab-b, 21-> CLU-v
			for (int cp_index : export_cps) {
				vector<Mat> concat_export;

				vector<Mat> circle_vect;
				vector<Mat> centaur_vect;

				Mat centaur_shower_img = draw::ext_fig_shower(imgOut, centaur_mask, cp_index);
				centaur_vect.push_back(centaur_shower_img);
				Mat circle_shower_img = draw::ext_fig_shower(imgOut, circle_mask, cp_index);
				circle_vect.push_back(circle_shower_img);

				for (int i = 0; i < img_quads.size(); i++) {
					Mat centaur_shower_img = draw::ext_fig_shower(img_quads[i], centaur_quads[i], cp_index);
					centaur_vect.push_back(centaur_shower_img);
					Mat circle_shower_img = draw::ext_fig_shower(img_quads[i], circle_quads[i], cp_index);
					circle_vect.push_back(circle_shower_img);
				}

			
			//	export_pics.push_back(centaur_shower_img);
			//	export_pics.push_back(circle_shower_img);

				//Batch resize - everything will be the dimentions of the img_quads[0] mat
				int rows_std = img_quads[0].rows;
				int cols_std = img_quads[0].cols;

				for (int i = 0; i < circle_vect.size(); i++) {
					resize(circle_vect[i], circle_vect[i], Size(cols_std, rows_std));
					resize(centaur_vect[i], centaur_vect[i], Size(cols_std, rows_std));
				}

				Mat circle_concat, centaur_concat, final_concat;

				vconcat(circle_vect, circle_concat);
				vconcat(centaur_vect, centaur_concat);

				vector<Mat> combos = { centaur_concat,circle_concat };
				hconcat(combos, final_concat);

				concat_export.push_back(final_concat);


				vector<Mat> h_concats;
				for (int i = 0; i < circle_vect.size(); i++) {
					Mat vCAT;
					vector<Mat> vCAT_vec = { centaur_vect[i],circle_vect[i] };
					vconcat(vCAT_vec, vCAT);
					Mat export_fig = vCAT.clone();
					concat_export.push_back(export_fig);

				}

				cp_Mat_export.push_back(concat_export);
	
			}

			vector<Mat> shuffle_pics;

			//Export vector<Mat> to pics vector
			//First just the summary photos
			for (int i = 0; i < cp_Mat_export.size(); i++) {

				shuffle_pics.push_back(cp_Mat_export[i][0]);

			}
			//Then the remaining four photos of the [0] colorspace

			for (int j = 1; j < cp_Mat_export[0].size(); j++) {
				shuffle_pics.push_back(cp_Mat_export[0][j]);

			}
	

			for (Mat fig : export_pics) {
				shuffle_pics.push_back(fig); if (shower_switch) tst::show(fig);
			}

			export_pics = shuffle_pics;

		}

		//Look at custom ROI
		cv::Point2f ROI_center = cv::Point2f(ROI_x, ROI_y);
		float ROI_radius = ROI_size;

		Mat ROI_out = roi::ROI_crop(imgOut, ROI_center.x, ROI_center.y, ROI_radius, ROI_radius, scaleOut, originOut, test_type);
		Mat circleMask_out = roi::ROI_crop(circle_mask, ROI_center.x, ROI_center.y, ROI_radius, ROI_radius, scaleOut, originOut, test_type);
		Mat centaurMask_out = roi::ROI_crop(centaur_mask, ROI_center.x, ROI_center.y, ROI_radius, ROI_radius, scaleOut, originOut, test_type);

		//Alignment Grid
		if (0) {
			for (float x = -10.25; x < 10; x += 0.5) {
				for (float y = -10.25; y < 10; y += 0.5) {

					ROI_center = cv::Point2f(x, y);

					int x_center_scale = int(ROI_center.x*scaleOut + originOut.x);
					int y_center_scale = int(ROI_center.y*scaleOut + originOut.y);
					int w_scale = int(scaleOut*ROI_radius);
					cv::rectangle(imgOut, cv::Point(x_center_scale - w_scale / 2., y_center_scale - w_scale / 2.), cv::Point(x_center_scale + w_scale / 2., y_center_scale + w_scale / 2.), Scalar(0, 0, 255), 2);


				}
			}
			circle(imgOut, originOut, 15, Scalar(0, 255, 0), -1);
		}

		if (shower_switch) tst::show(imgOut);


		int x_center_scale = int(ROI_center.x*scaleOut + originOut.x);
		int y_center_scale = int(ROI_center.y*scaleOut + originOut.y);
		int w_scale = int(scaleOut*ROI_radius);
		cv::rectangle(imgOut, cv::Point(x_center_scale - w_scale / 2., y_center_scale - w_scale / 2.), cv::Point(x_center_scale + w_scale / 2., y_center_scale + w_scale / 2.), Scalar(0, 0, 255), 2);


		Mat ROI_circle_shower = draw::ext_fig_shower(ROI_out, circleMask_out, 18);

		Mat ROI_centaur_shower = draw::ext_fig_shower(ROI_out, centaurMask_out, 18);

		vector<Mat> vROIcat = { ROI_centaur_shower ,ROI_circle_shower };
		Mat vROI;
		vconcat(vROIcat, vROI);

		vector<Mat> shuffle_pics2;
		shuffle_pics2.push_back(export_pics[0]);
		shuffle_pics2.push_back(vROI);
		shuffle_pics2.push_back(imgOut);

		resize(ROI_out, ROI_out, Size(500, 500));
		shuffle_pics2.push_back(ROI_out);
		for (Mat fig : export_pics) shuffle_pics2.push_back(fig);

		
		
		//Edits for black-on-white UI
		if (find(test_type.begin(), test_type.end(), 1) != test_type.end()) {
			export_pics.push_back(img);
			export_pics.push_back(imgOut);
		}
		else export_pics = shuffle_pics2;
		read_result_reference = 1;
		user_comments = data_output;

		


		if (find(test_type.begin(), test_type.end(), -2) != test_type.end()) {
			//check how well ROI_out matches with reference, based on an otsu thresholded HSV-V
			Mat ROI_test = color::makeColorspace_single(ROI_out, 12);
			Mat ROI_test2;
			if (shower_switch) tst::show(ROI_test);
			double ot_thresh_test = threshold(ROI_test, ROI_test, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			if (line_by_line_check) cout << "Thresh:" << ot_thresh_test << endl;
			//threshold(ROI_test, ROI_test, ot_thresh_test, 255, CV_THRESH_BINARY);
			medianBlur(ROI_test, ROI_test, 11);
			if (shower_switch) tst::show(ROI_test);
			//imwrite("standard binary ROI.jpg", ROI_test);
			Mat ROI_standard = imread("Resources//Templates//JB Binary Template.jpg", 0);
			threshold(ROI_standard, ROI_standard, 125, 255, CV_THRESH_BINARY);
			cv::Mat match_result;
			matchTemplate(ROI_test, ROI_standard, match_result, CV_TM_CCORR_NORMED);
			double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
			minMaxLoc(match_result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
			if (line_by_line_check) cout << "Min value: " << maxVal << endl;

			export_pics.push_back(ROI_test);
			user_comments += "matchvalue:" + to_string(maxVal);
		}

		

		return export_pics;
	}

	void feature_finder(Mat imageMat, vector<Mat> masks, Mat &imageOutput, Mat &maskOutput, cv::Point2f &originOutput, float &scaleOutput, vector<Mat> &export_pics, string &data_output, int &exception_output, string mask_path, vector<int> test_type) {
		bool shower_switch = 0;
		bool line_by_line_check = 0;
		if (shower_switch) tst::show(imageMat);
		Mat imageMat_draw = imageMat.clone();

		vector<Mat> internal_shower;
		internal_shower.push_back(imageMat);

		vector<Mat> internal_shower2;

		//Extract in various colorspaces
		vector<int> colorspace_indices = { 15,18 }; //15 -> HLS-S; 18-> Lab-b 
		double min_match = 999;
		vector<cv::Point> winner_cnt;
		bool fat_switch;
		for (int cp_index : colorspace_indices) {
			Mat img_ext = color::makeColorspace_single(imageMat, cp_index);

			//otsu
			double thresh_val;
			if (cp_index == 15) thresh_val = 240;
			else if (cp_index == 18) thresh_val = 138;
			threshold(img_ext, img_ext, thresh_val, 255, THRESH_BINARY);

			medianBlur(img_ext, img_ext, 3); //Same med blur value as was used in CG

			//Morphological Transformations
			//img_ext = color::morph(img_ext, 5, 3);
			img_ext = color::morph(img_ext, 11, 4);

			//Find contours
			std::vector<std::vector<cv::Point> > contours;
			std::vector<Vec4i> hierarchy;

			cv::findContours(img_ext, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			//drawContours(imageMat, contours, -1, Scalar(0, 255, 0), 10);

			cvtColor(img_ext, img_ext, CV_GRAY2BGR);
			//internal_shower.push_back(img_ext);


			std::vector<std::vector<cv::Point> > contours_big;
			if (contours.size()>0) contours = contour::size_threshold(contours, 50000);

			if (contours.size() > 0) {
				Mat contour_draw = contour::draw_cnt(contours, img_ext);

				internal_shower.push_back(contour_draw);
				if (shower_switch) tst::show(contour_draw);
				//imwrite("export.jpg", contour_draw);

				//Match to contour
				string mask_name_fat = "JB_BC_Fat_0";
				string mask_name = "JB_BC_0";
				map<string, vector<cv::Point>> mask_map = contour::mask_map_maker({ mask_name,mask_name_fat }, mask_path);

				vector<double> match_values;
				//Match to fat contour
				vector<vector<cv::Point>> contour_match = contour::match_contours(contours, mask_map[mask_name_fat], match_values);
				double fat_match, regular_match;
				fat_match = match_values[0];
				if (match_values[0] < min_match) {
					min_match = match_values[0];
					winner_cnt = contour_match[0];
					fat_switch = true;

				}

				//Match to regular contour
				contour_match = contour::match_contours(contours, mask_map[mask_name], match_values);
				regular_match = match_values[0];
				if (match_values[0] < min_match) {
					min_match = match_values[0];
					winner_cnt = contour_match[0];
					fat_switch = false;

				}
				if (line_by_line_check) cout << "Fat Match=" << fat_match << ", Regular Match=" << regular_match << endl;
			}
		}

		if (line_by_line_check) cout << "Min match:" <<min_match << endl;

		//If min_match is greater than 1, throw an exception
		if (min_match > 0.25) {
			exception_output = -1;
			return;
		}

		double winner_cnt_area = contourArea(winner_cnt);
		int max_cnt_area_thresh = int(winner_cnt_area*0.95);
		if (line_by_line_check) cout << "Winner Area: " << winner_cnt_area << endl;

		//Stage 2 of Feature Finding - Internal elements
		Mat centaur_mask = contour::draw_cnt(winner_cnt, imageMat);
		//centaur_mask = color::morph(centaur_mask, 5, 1);
		if (shower_switch) tst::show(centaur_mask);
		Mat1b imgOut_ext = color::makeColorspace_single(imageMat, 12); //Index 12 is HSV-V
		double otsu_threshold = color::otsu_8u_with_mask(imgOut_ext, centaur_mask);
		Mat1b centaur_otsu_mask; threshold(imgOut_ext, centaur_otsu_mask, otsu_threshold, 255, THRESH_BINARY);
		bitwise_not(centaur_otsu_mask, centaur_otsu_mask);
		bitwise_and(centaur_otsu_mask, centaur_mask, centaur_otsu_mask);

		if (shower_switch) tst::show(centaur_otsu_mask);

		//Clean up binary image
		/*centaur_otsu_mask = color::morph(centaur_otsu_mask, 5, 3);
		centaur_otsu_mask = color::morph(centaur_otsu_mask, 5, 4);
*/
		medianBlur(centaur_otsu_mask, centaur_otsu_mask, 5);

		if (shower_switch) tst::show(centaur_otsu_mask);

		//Find contours
		std::vector<std::vector<cv::Point> > contours;
		std::vector<Vec4i> hierarchy;

		cv::findContours(centaur_otsu_mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		//Keep only white contours
		vector<vector<cv::Point>> white_contours = contour::white_contour_extraction(contours, hierarchy, centaur_otsu_mask);

		white_contours = contour::size_threshold(white_contours, 100, max_cnt_area_thresh);

		Mat cnt_draw3 = contour::draw_cnt(white_contours, imageMat);

		for (vector<cv::Point> cnt : white_contours) if (line_by_line_check) cout << contourArea(cnt) << ",";

		if (line_by_line_check) cout << endl;

		internal_shower2.push_back(cnt_draw3);
		if (shower_switch) tst::show(cnt_draw3);

		////Filter out top-hirearchy contours
		//std::vector<std::vector<cv::Point>> filtered_contours;
		//for (int i = 0; i < contours.size(); i++) {
		//	if (hierarchy[i][3] >= 1) {
		//		filtered_contours.push_back(contours[i]);
		//	}
		//}




		data_output = "minmatch:" + to_string(min_match) + ",winnerarea:" + to_string(contourArea(winner_cnt))+",";

		Mat cnt_draw = contour::draw_cnt({ winner_cnt }, imageMat);
			
		cvtColor(cnt_draw, cnt_draw, CV_GRAY2BGR);

		cv::Point2f mec_center; float mec_radius;
			minEnclosingCircle(winner_cnt, mec_center, mec_radius);

			Vec4f lin;
			fitLine(winner_cnt, lin, CV_DIST_L2,0,10,0.01);

			line(cnt_draw, cv::Point(lin[2] - lin[0] * 1000, lin[3] - lin[1] * 1000), cv::Point(lin[2] + lin[0]*1000, lin[3] + lin[1] * 1000), Scalar(255, 0, 0), 10);
			circle(cnt_draw, mec_center, 1, Scalar(0, 255, 0), 10);
			circle(cnt_draw, mec_center, mec_radius, Scalar(0, 0, 255), 10);

			vector<vector<cv::Point>> winner_cnt_draw = { winner_cnt };
			drawContours(imageMat_draw, winner_cnt_draw, -1, Scalar(255, 0, 255), 15);
			line(imageMat_draw, cv::Point(lin[2] - lin[0] * 1000, lin[3] - lin[1] * 1000), cv::Point(lin[2] + lin[0] * 1000, lin[3] + lin[1] * 1000), Scalar(255, 0, 0), 10);
			circle(imageMat_draw, mec_center, 10, Scalar(0, 255, 0), -1);
			circle(imageMat_draw, mec_center, mec_radius, Scalar(0, 0, 255), 10);
			internal_shower.push_back(imageMat_draw);
			internal_shower2.push_back(imageMat_draw);

			////For v0 algo:
			//Scale - radius of enclosing circle
			//Origin - center of enclosing circle
			//Rot - line of best fit

			scaleOutput = mec_radius;
			originOutput = cv::Point(int(mec_radius),int(mec_radius));//The origin is the center of the otuput cropped image.

			float angle = atan(float(lin[1]) / float(lin[0]))*(float(180)/float(3.14159));

			//Define mask
			Mat imageMatmask = contour::draw_cnt(winner_cnt, imageMat);

			//Rotate Image and Mask
			Mat imageMat_rot,imageMatmask_rot;
			Mat rotation_kernel = getRotationMatrix2D(mec_center, angle, 1.0);
			warpAffine(imageMat, imageMat_rot, rotation_kernel, Size(imageMat.cols, imageMat.rows));
			warpAffine(imageMatmask, imageMatmask_rot, rotation_kernel, Size(imageMat.cols, imageMat.rows));
			threshold(imageMatmask_rot, imageMatmask_rot, 125, 255, THRESH_BINARY);

			//Crop to enclosing circle dimentions
			Rect crop_rect = Rect(mec_center.x - mec_radius, mec_center.y - mec_radius, mec_radius * 2, mec_radius * 2);
			Mat imageMat_crop = Mat(imageMat_rot, crop_rect).clone();
			Mat imageMatmask_crop = Mat(imageMatmask_rot, crop_rect).clone();

			imageOutput = imageMat_crop;
			maskOutput = imageMatmask_crop;

			export_pics.push_back(imageMat_draw);

		Mat hMat;
		hconcat(internal_shower, hMat);

		//resize(hMat, hMat, Size(0, 0),0.2,0.2);

		Mat hMat2;
		hconcat(internal_shower2, hMat2);

		//resize(hMat2, hMat2, Size(0, 0), 0.2, 0.2);

		export_pics.push_back(hMat);
		export_pics.push_back(hMat2);


		//Stage 3 of feature finding - Improved accuracy via internal elements
		if (line_by_line_check) cout << "Stage 3" << endl;
		Mat internal_elements = contour::draw_cnt(white_contours, imageMat,true);
		internal_elements = color::makeColorspace_single(internal_elements, 12);


		// temporarily get rid of keeping white contours
		internal_elements = centaur_otsu_mask;

		//Rotate
		Mat internal_elements_rot,centaur_mask_rot;
		rotation_kernel = getRotationMatrix2D(mec_center, angle, 1.0);
		warpAffine(internal_elements, internal_elements_rot, rotation_kernel, Size(imageMat.cols, imageMat.rows));
		threshold(internal_elements_rot, internal_elements_rot, 125, 255, THRESH_BINARY);
 
		vector<Rect2f> internal_crops;
		float height_boost = 0.5;
		internal_crops.push_back(Rect2f(0.0, -0.5-height_boost, 0.5, 0.5 + height_boost)); //Head
		internal_crops.push_back(Rect2f(-0.1, -0.3 - height_boost, 0.4, 0.6 + height_boost)); //Body
		internal_crops.push_back(Rect2f(-1.0-0.3, -0.3 - height_boost-0.2, 0.7+0.6, 0.7 + height_boost+0.8)); //Tail

		//Make map of masks
		vector<string> reference_masks = { "JB_BC_InnerCnt_Head_0" , "JB_BC_InnerCnt_Body_0" ,"JB_BC_InnerCnt_Tail_0" };
		map<string, vector<cv::Point>> internal_masks = contour::mask_map_maker(reference_masks, mask_path);

		//Save working scale
		data_output += "roughscale:" + to_string(mec_radius) + ",";

		//Set up specific area thresholds
		vector<double> normalized_scale_vec = { 1.25,5.29,3.84 };

		if (shower_switch) tst::show(internal_elements_rot);
		map<string,cv::Point> internal_centers;
		for (int i = 0; i < internal_crops.size();i++){ 

			Rect2f crop_rectangle = internal_crops[i];

			Mat ROI_out = roi::ROI_crop(internal_elements_rot, crop_rectangle.x+crop_rectangle.width/2, crop_rectangle.y + crop_rectangle.height / 2, crop_rectangle.width, crop_rectangle.height, mec_radius, cv::Point(int(mec_center.x), int(mec_center.y)), test_type);
			if (shower_switch) tst::show(ROI_out);

			//Find contours
			std::vector<std::vector<cv::Point> > ROI_contours;
			std::vector<Vec4i> ROI_hierarchy;

			cv::findContours(ROI_out, ROI_contours, ROI_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			if (line_by_line_check) cout << "Filtering internal contours by size..." << endl;

			double specific_size_threshold = normalized_scale_vec[i] * double(mec_radius);

			ROI_contours = contour::size_threshold(ROI_contours, specific_size_threshold);

			//If you're looking for the head contour, don't accept contours that are on the edges.
			if (i == 0 || i==2) {
				vector<vector<cv::Point>> filtered_contours;
				int buffer = 3;
				for (vector<cv::Point> cnt : ROI_contours) {
					int top, bot, left, right;
					contour::extreme_points(cnt, top, bot, left, right);
					if (top < buffer || left < buffer) continue;
					if (bot > ROI_out.rows - buffer || right > ROI_out.cols - buffer) continue;
					filtered_contours.push_back(cnt);
				}
				ROI_contours = filtered_contours;
			}


			if (line_by_line_check) cout << "Remianing contour count: " << ROI_contours.size() << endl;

			if (ROI_contours.size() == 0) {
				exception_output = -2;
				return;
			}

			Mat filtered_internal_cnt = contour::draw_cnt(ROI_contours, ROI_out, true);

			vector<double> ROI_match_output;
			vector<vector<cv::Point>> winner_internal = contour::match_contours(ROI_contours, internal_masks[reference_masks[i]], ROI_match_output);



			cv::Point ROI_center = contour::center(winner_internal[0]);

			cvtColor(ROI_out, ROI_out, COLOR_GRAY2BGR);

			if (shower_switch) tst::show(filtered_internal_cnt);
			drawContours(filtered_internal_cnt, winner_internal, -1, Scalar(255, 0, 0), 5);
			circle(ROI_out, ROI_center, 5, Scalar(0, 255, 0), -1);
			
			if (shower_switch) tst::show(filtered_internal_cnt);
			if (line_by_line_check) cout << "Internal min match: " << ROI_match_output[0] << endl;

			data_output += "roi" + to_string(i) + "cmatch:" + to_string(ROI_match_output[0]) + ",";

			data_output += "roi" + to_string(i) + "area:" + to_string(contourArea(winner_internal[0])) + ",";

			Rect ROI_scalar=roi::ROI_scale(crop_rectangle.x + crop_rectangle.width / 2, crop_rectangle.y + crop_rectangle.height / 2, crop_rectangle.width, crop_rectangle.height, mec_radius, cv::Point(int(mec_center.x), int(mec_center.y)), test_type);

			cv::Point export_point = cv::Point(ROI_scalar.x + ROI_center.x, ROI_scalar.y + ROI_center.y);
			data_output += "roi" + to_string(i) + "xcenter:" + to_string(export_point.x)+",";
			data_output += "roi" + to_string(i) + "ycenter:" + to_string(export_point.y) + ",";
			internal_centers[reference_masks[i]]=export_point;
		}

		vector<cv::Point> tridot;
		cvtColor(internal_elements_rot, internal_elements_rot, COLOR_GRAY2BGR);
		for (string str : reference_masks) {
			circle(internal_elements_rot, internal_centers[str], 10, Scalar(0, 255, 0), -1);
			tridot.push_back(internal_centers[str]);
		}
		if (shower_switch) tst::show(internal_elements_rot);

		cv::Point tridot_center = contour::center(tridot);

		circle(internal_elements_rot, tridot_center, 30, Scalar(0, 0, 255), -1);

		Vec4f lin2;
		fitLine(tridot, lin2, CV_DIST_L2, 0, 10, 0.01);
		//lin2 = { float(internal_centers["JB_BC_InnerCnt_Body_0"].x) - float(internal_centers["JB_BC_InnerCnt_Tail_0"].x), float(internal_centers["JB_BC_InnerCnt_Body_0"].y) - float(internal_centers["JB_BC_InnerCnt_Tail_0"].y),float(internal_centers["JB_BC_InnerCnt_Tail_0"].x),float(internal_centers["JB_BC_InnerCnt_Tail_0"].y) };

		line(internal_elements_rot, cv::Point(lin2[2] - lin2[0] * 1000, lin2[3] - lin2[1] * 1000), cv::Point(lin2[2] + lin2[0] * 1000, lin2[3] + lin2[1] * 1000), Scalar(255, 0, 0), 10);

		if (shower_switch) tst::show(internal_elements_rot);


		//Save improved feature fing
		scaleOutput = sqrt(float(contourArea(tridot)));
		originOutput = tridot_center;

		float angle2 = atan(float(lin2[1]) / float(lin2[0]))*(float(180) / float(3.14159))+angle;

		//Rotate Image and Centaur Mask
		Mat imageMat_rot_fin, centaurmask_rot_fin;
		Mat rotation_kernel_fin = getRotationMatrix2D(mec_center, angle2, 1.0);
		warpAffine(imageMat, imageMat_rot_fin, rotation_kernel_fin, Size(imageMat.cols, imageMat.rows));
		warpAffine(centaur_mask, centaurmask_rot_fin, rotation_kernel, Size(imageMat.cols, imageMat.rows));
		threshold(centaurmask_rot_fin, centaurmask_rot_fin, 125, 255, THRESH_BINARY);

		imageOutput = imageMat_rot_fin;
		maskOutput = centaurmask_rot_fin;

		export_pics.push_back(imageMat_draw);

		data_output += "ffcenterx:" + to_string(originOutput.x) + ",";
		data_output += "ffcentery:" + to_string(originOutput.y) + ",";
		data_output += "ffrot:" + to_string(angle2) + ",";
		data_output += "ffscale:" + to_string(scaleOutput) + ",";

		Mat im_show = imageMat_rot_fin.clone();
		rectangle(im_show, Rect(tridot_center.x, tridot_center.y, int(scaleOutput*1), int(scaleOutput*1)), Scalar(0, 255, 0), 50);

		if (shower_switch) tst::show(im_show);

		return;
	}
}

//////////////////////////////////////// NEXT STEPS /////////////////////////////////////////
///// I need to enable to code to define features based on only finding two out of the three reference contours.
