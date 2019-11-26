#include "pch.h"
#include "Face_detector.h"

void CallBackFunc(int event, int x, int y, int flags, void* userdata);

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
	//Given a vector of streams, we try to find a depth stream and another stream to align depth with.
	//We prioritize color streams to make the view look better.
	//If color is not available, we take another stream that (other than depth)
	rs2_stream align_to = RS2_STREAM_ANY;
	bool depth_stream_found = false;
	bool color_stream_found = false;
	for (rs2::stream_profile sp : streams)
	{
		rs2_stream profile_stream = sp.stream_type();
		if (profile_stream != RS2_STREAM_DEPTH)
		{
			if (!color_stream_found)         //Prefer color
				align_to = profile_stream;

			if (profile_stream == RS2_STREAM_COLOR)
			{
				color_stream_found = true;
			}
		}
		else
		{
			depth_stream_found = true;
		}
	}

	if (!depth_stream_found)
		throw std::runtime_error("No Depth stream available");

	if (align_to == RS2_STREAM_ANY)
		throw std::runtime_error("No stream found to align with Depth");

	return align_to;
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
	for (auto&& sp : prev)
	{
		//If previous profile is in current (maybe just added another)
		auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
		if (itr == std::end(current)) //If it previous stream wasn't found in current
		{
			return true;
		}
	}
	return false;
}

Face_detector::Face_detector(const unsigned int x, const unsigned int y, const std::string name) :
	size_x(x),
	size_y(y),
	window_name(name),
	ROI_file("ROIs/ROIs.txt", std::ios::app),
	val_ROI_file("ROIs/val_ROIs.txt", std::ios::app),
	test_ROI_file("ROIs/test_ROIs.txt", std::ios::app)
{
	if (!ROI_file.good() || !test_ROI_file.good()) {
		std::cerr << "Cannot open ROI file" << std::endl;
		std::exit(-1);
	}
	detector = dlib::get_frontal_face_detector();
	std::cout << "Done." << std::endl;
	predictorWeights = "shape_predictor_68_face_landmarks.dat";
	dlib::deserialize(predictorWeights) >> shapePredictor;
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8);
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16);
	cfg.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8);
	profile = input.start(cfg);
	align_to = find_stream_to_align(profile.get_streams());
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name, size_x, size_y);
	cv::setMouseCallback(window_name, CallBackFunc, this);

	std::ifstream master("master.txt");
	if (!master.good()) {
		std::cerr << "Cannot open master file" << std::endl;
		std::exit(-1);
	}
	master >> saved_train;
	master >> saved_val;
	master >> saved_test;
	master.close();

	std::cout << "Current image count: " << std::to_string(saved_train) << " training, " << std::to_string(saved_val) << " validation, " << std::to_string(saved_test) << " test" << std::endl;
}

Face_detector::~Face_detector() {
	for (Face_display* f : face_windows) {
		delete f;
	}
	std::ofstream master("master.txt");
	if (!master.good()) {
		std::cerr << "Cannot open master file" << std::endl;
		std::exit(-1);
	}
	master << saved_train << std::endl;
	master << saved_val << std::endl;
	master << saved_test << std::endl;
	master.close();
	ROI_file.close();
	test_ROI_file.close();
}

void Face_detector::process_input() {

	rs2::align align(align_to);
	if (profile_changed(input.get_active_profile().get_streams(), profile.get_streams()))
	{
		//If the profile was changed, update the align object, and also get the new device's depth scale
		profile = input.get_active_profile();
		align_to = find_stream_to_align(profile.get_streams());
		align = rs2::align(align_to);
	}

	rs2::frameset data = input.wait_for_frames();
	auto processed = align.process(data);

	rs2::depth_frame depth = processed.get_depth_frame();
	rs2::video_frame col = processed.get_color_frame();
	rs2::device dev = profile.get_device();
	rs2::depth_sensor ds = dev.query_sensors().front().as<rs2::depth_sensor>();
	rs2::frame ir = processed.first(RS2_STREAM_INFRARED);

	//Filter block for depth stream denoising
	rs2::decimation_filter dec_fil;
	dec_fil.set_option(RS2_OPTION_FILTER_MAGNITUDE, 3.0f);
	rs2::spatial_filter spat_fil;
	spat_fil.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2.0f);
	spat_fil.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5f);
	spat_fil.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.0f);
	rs2::temporal_filter temp_fil;
	temp_fil.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
	temp_fil.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.0f);

	rs2::frame filtered = depth;
	//filtered = dec_fil.process(depth); // Reduces resoltuion
	filtered = depth;
	filtered = spat_fil.process(filtered);
	filtered = temp_fil.process(filtered);
	rs2::frame col_depth = filtered.apply_filter(color_map);


	float scale = ds.get_depth_scale();
	cv::Mat image(cv::Size(col.get_width(), col.get_height()), CV_8UC3, (void*)col.get_data(), cv::Mat::AUTO_STEP);
	cv::Mat im_small;
	cv::resize(image, im_small, cv::Size(320, 240)); // kicsinyítés, hogy gyorsabb legyen,
	cvtColor(im_small, im_small, cv::COLOR_BGR2GRAY); // grayscale, hogy gyorsabb legyen
	dlib::cv_image<uchar> dlib_img(im_small);
	std::vector<dlib::rectangle> faces = detector(dlib_img);
	while (face_windows.size() != faces.size()) {
		if (face_windows.size() < faces.size()) {
			face_windows.push_back(new Face_display(600, 300, "Face " + std::to_string(face_windows.size() + 1)));
		}
		if (face_windows.size() > faces.size()) {
			delete face_windows.back();
			face_windows.erase(face_windows.begin() + face_windows.size() - 1);
		}
	}
	
	cv::Mat depth_data(cv::Size(filtered.as<rs2::depth_frame>().get_width(), filtered.as<rs2::depth_frame>().get_height()), CV_16UC1, (void*)filtered.as<rs2::depth_frame>().get_data(), cv::Mat::AUTO_STEP);
	cv::Mat depth_image(cv::Size(col_depth.as<rs2::video_frame>().get_width(), col_depth.as<rs2::video_frame>().get_height()), CV_8UC3, (void*)col_depth.as<rs2::video_frame>().get_data(), cv::Mat::AUTO_STEP);
	cv::Mat col_image(cv::Size(col.as<rs2::video_frame>().get_width(), col.as<rs2::video_frame>().get_height()), CV_8UC3, (void*)col.as<rs2::video_frame>().get_data(), cv::Mat::AUTO_STEP);
	cv::Mat display = col_image.clone();
	cv::Mat ir_image(cv::Size(ir.as<rs2::video_frame>().get_width(), ir.as<rs2::video_frame>().get_height()), CV_8UC1, (void*)ir.as<rs2::video_frame>().get_data(), cv::Mat::AUTO_STEP);

	if (faces.size() != 0)
	{
		for (int i = 0; i < faces.size(); i+=2) {
			cv::Rect faceRect = cv::Rect(cv::Point2i(faces[i].left(), faces[i].top()), cv::Point2i(faces[i].right() + 1, faces[i].bottom() + 1));
			float scale_x = (float)depth_image.cols / 320.0;
			float scale_y = (float)depth_image.rows / 240.0;
			float size_factor = 1.0f;
			faceRect.width = faceRect.width * scale_x * size_factor;
			faceRect.height = faceRect.height * scale_y * size_factor;
			faceRect.x = faceRect.x * scale_x;
			faceRect.y = faceRect.y * scale_y;

			if (faceRect.x < 0) faceRect.x = 0;
			if (faceRect.y < 0) faceRect.y = 0;
			if (faceRect.x + faceRect.width >= depth_image.cols) faceRect.x = depth_image.cols - faceRect.width;
			if (faceRect.y + faceRect.height >= depth_image.rows) faceRect.y = depth_image.rows - faceRect.height;
			cv::rectangle(display, faceRect, cv::Scalar(0, 0, 0), 4);

			cv::Mat d_crop = depth_data(faceRect);
			cv::Mat d_image_crop = depth_image(faceRect);
			cv::Mat image_crop = col_image(faceRect);
			cv::hconcat(image_crop, d_image_crop, image_crop);
			face_windows[i]->show_face(image_crop);
			if (save || video) {
				save = false;
				if (test) {
					cv::imwrite("color/test_col" + std::to_string(saved_test) + ".png", col_image);
					cv::imwrite("depth_data/test_depth" + std::to_string(saved_test) + ".png", depth_data);
					cv::imwrite("depth_color/test_col_depth" + std::to_string(saved_test) + ".png", depth_image);
					cv::imwrite("IR/test_ir" + std::to_string(saved_test) + ".png", ir_image);
					test_ROI_file << faceRect.x << ' ' << faceRect.y << ' ' << faceRect.width << ' ' << faceRect.height << std::endl;
					std::cout << "Saved test" << std::to_string(saved_test) << std::endl;
					++(saved_test);
				}
				else {
					if (val) {
						cv::imwrite("color/val_col" + std::to_string(saved_val) + ".png", col_image);
						cv::imwrite("depth_data/val_depth" + std::to_string(saved_val) + ".png", depth_data);
						cv::imwrite("depth_color/val_col_depth" + std::to_string(saved_val) + ".png", depth_image);
						cv::imwrite("IR/vali_ir" + std::to_string(saved_val) + ".png", ir_image);
						val_ROI_file << faceRect.x << ' ' << faceRect.y << ' ' << faceRect.width << ' ' << faceRect.height << std::endl;
						std::cout << "Saved val" << std::to_string(saved_val) << std::endl;
						++(saved_val);
					}
					else {
						cv::imwrite("color/col" + std::to_string(saved_train) + ".png", col_image);
						cv::imwrite("depth_data/depth" + std::to_string(saved_train) + ".png", depth_data);
						cv::imwrite("depth_color/col_depth" + std::to_string(saved_train) + ".png", depth_image);
						cv::imwrite("IR/train_ir" + std::to_string(saved_train) + ".png", ir_image);
						ROI_file << faceRect.x << ' ' << faceRect.y << ' ' << faceRect.width << ' ' << faceRect.height << std::endl;
						std::cout << "Saved train" << std::to_string(saved_train) << std::endl;
						++(saved_train);
					}
				}

			}
		}
	}
	if (save) save = false;
	cv::imshow(window_name, display);

}

Face_detector::Face_display::Face_display(const unsigned int x, const unsigned int y, const std::string name) : size_x(x), size_y(y), window_name(name) {
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name, size_x, size_y);
}

Face_detector::Face_display::~Face_display() {
	cv::destroyWindow(window_name);
}

void Face_detector::Face_display::show_face(cv::Mat& image) {
	cv::imshow(window_name, image);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		(reinterpret_cast<Face_detector*>(userdata))->save = true;
	}

	if (event == cv::EVENT_RBUTTONDOWN)
	{
		if (!(reinterpret_cast<Face_detector*>(userdata))->test) {
			(reinterpret_cast<Face_detector*>(userdata))->val = !(reinterpret_cast<Face_detector*>(userdata))->val;
			if ((reinterpret_cast<Face_detector*>(userdata))->val) {
				std::cout << "Saving to validation dataset" << std::endl;
			}
			else {
				std::cout << "Saving to training dataset" << std::endl;
			}
		}
	}

	if (event == cv::EVENT_RBUTTONDBLCLK)
	{
		(reinterpret_cast<Face_detector*>(userdata))->test = !(reinterpret_cast<Face_detector*>(userdata))->test;
		if ((reinterpret_cast<Face_detector*>(userdata))->test) {
			std::cout << "Saving to testing dataset" << std::endl;
		}
		else {
			if ((reinterpret_cast<Face_detector*>(userdata))->val) {
				std::cout << "Saving to validation dataset" << std::endl;
			}
			else {
				std::cout << "Saving to training dataset" << std::endl;
			}
		}
	}

	if (event == cv::EVENT_MBUTTONDOWN) {
		(reinterpret_cast<Face_detector*>(userdata))->video = !(reinterpret_cast<Face_detector*>(userdata))->video;
	}
}