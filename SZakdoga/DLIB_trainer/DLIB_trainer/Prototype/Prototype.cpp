#include "pch.h"
#include "helper.hpp"
#include "stdlib.h"
#include "fstream"
#include "math.h"
#include "chrono"

void build(std::string filepath_base, bool test) {
	std::string filepath;
	std::cout << "Building XML file" << std::endl;
	filepath = filepath_base + ((test) ? "\\depth_stream_testing.xml" : "\\depth_stream_training.xml");
	std::ofstream fout(filepath);
	fout << "<?xml version='1.0' encoding='ISO-8859-1'?>" << std::endl;
	fout << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>" << std::endl;
	fout << "<dataset>" << std::endl;
	fout << "<name>imglab dataset</name>" << std::endl;
	fout << "<comment>Created by imglab tool.</comment>" << std::endl;
	fout << "<images>" << std::endl;

	std::cout << filepath_base + "\\master.txt" << std::endl;
	std::ifstream master(filepath_base + "\\master.txt");
	if (!master.good()) {
		std::cout << "Cannot open master file!" << std::endl;
		exit(-1);
	}
	int num_train, num_test;
	master >> num_train >> num_test;
	master.close();
	std::string ROI_filepath = filepath_base + ((test) ? "\\ROIs\\test_ROIs.txt" : "\\ROIs\\ROIs.txt");
	std::ifstream ROIs(ROI_filepath);
	if (!ROIs.good()) {
		std::cout << "Cannot open ROI file!" << std::endl;
		exit(-1);
	}
	int lim = ((test) ? num_test : num_train);
	for (int i = 0; i < lim; ++i) {
		if (test) {
			fout << "\t<image file='.\\depth_color\\test_col_depth" + std::to_string(i) + ".png'>" << std::endl;
		}
		else {
			fout << "\t<image file='.\\depth_color\\col_depth" + std::to_string(i) + ".png'>" << std::endl;
		}
		int x, y, w, h;
		ROIs >> x >> y >> w >> h;
		fout << "\t\t<box top='" + std::to_string(y) + "' left='" + std::to_string(x) + "' width='" + std::to_string(w) + "' height='" + std::to_string(h) + "'/>" << std::endl;
		fout << "\t</image>" << std::endl;
	}
	fout << "</images>" << std::endl;
	fout << "</dataset>" << std::endl;
	fout.close();
}

void build_small(std::string filepath_base) {
	std::string filepath;
	std::cout << "Building XML file" << std::endl;
	filepath = filepath_base + "\\depth_stream_training_small.xml";
	std::ofstream fout(filepath);
	fout << "<?xml version='1.0' encoding='ISO-8859-1'?>" << std::endl;
	fout << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>" << std::endl;
	fout << "<dataset>" << std::endl;
	fout << "<name>imglab dataset</name>" << std::endl;
	fout << "<comment>Created by imglab tool.</comment>" << std::endl;
	fout << "<images>" << std::endl;

	std::cout << filepath_base + "\\master.txt" << std::endl;
	std::ifstream master(filepath_base + "\\master.txt");
	if (!master.good()) {
		std::cout << "Cannot open master file!" << std::endl;
		exit(-1);
	}
	int num_train, num_test;
	master >> num_train >> num_test;
	master.close();
	std::string ROI_filepath = filepath_base + "\\ROIs\\ROIs.txt";
	std::ifstream ROIs(ROI_filepath);
	if (!ROIs.good()) {
		std::cout << "Cannot open ROI file!" << std::endl;
		exit(-1);
	}
	int lim = num_train;
	std::vector<std::vector<int>> rects(lim, std::vector<int>(4));
	for (int i = 0; i < lim; ++i) {
		int x, y, w, h;
		ROIs >> x >> y >> w >> h;
		rects[i][0] = x; rects[i][1] = y; rects[i][2] = w; rects[i][3] = h;
	}
	std::vector<uint16_t> pics = {7, 79, 192, 276, 311, 381, 457, 534, 590, 695, 737, 870, 989, 1080, 1116, 1240, 1311, 1375, 1454, 1514, 1556, 1645, 1737, 1802, 1860, 1919, 1998, 2137};
	for (uint16_t i : pics) {
		for (int j = -7; j <= 7; ++j) {
			fout << "\t<image file='.\\depth_color\\col_depth" + std::to_string(i+j) + ".png'>" << std::endl;
			fout << "\t\t<box top='" + std::to_string(rects[i+j][0]) + "' left='" + std::to_string(rects[i + j][1]) + "' width='" + std::to_string(rects[i + j][2]) + "' height='" + std::to_string(rects[i + j][3]) + "'/>" << std::endl;
			fout << "\t</image>" << std::endl;
		}
	}
	fout << "</images>" << std::endl;
	fout << "</dataset>" << std::endl;
	fout.close();
}

void build_small2(std::string filepath_base) {
	std::string filepath;
	std::cout << "Building XML file" << std::endl;
	filepath = filepath_base + "\\depth_stream_training_small.xml";
	std::ofstream fout(filepath);
	fout << "<?xml version='1.0' encoding='ISO-8859-1'?>" << std::endl;
	fout << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>" << std::endl;
	fout << "<dataset>" << std::endl;
	fout << "<name>imglab dataset</name>" << std::endl;
	fout << "<comment>Created by imglab tool.</comment>" << std::endl;
	fout << "<images>" << std::endl;

	std::cout << filepath_base + "\\master.txt" << std::endl;
	std::ifstream master(filepath_base + "\\master.txt");
	if (!master.good()) {
		std::cout << "Cannot open master file!" << std::endl;
		exit(-1);
	}
	int num_train, num_test;
	master >> num_train >> num_test;
	master.close();
	std::string ROI_filepath = filepath_base + "\\ROIs\\ROIs.txt";
	std::ifstream ROIs(ROI_filepath);
	if (!ROIs.good()) {
		std::cout << "Cannot open ROI file!" << std::endl;
		exit(-1);
	}
	int lim = num_train;
	std::vector<std::vector<int>> rects(lim, std::vector<int>(4));
	for (int i = 0; i < lim; ++i) {
		int x, y, w, h;
		ROIs >> x >> y >> w >> h;
		rects[i][1] = x; rects[i][0] = y; rects[i][2] = w; rects[i][3] = h;
	}
	std::vector<uint16_t> start = {0, 518, 1102, 1615};
	std::vector<uint16_t> end = {72, 555, 1183, 1702};
	for (int i = 0; i < start.size(); ++i) {
		for (int j = start[i]; j < end[i]; ++j) {
			fout << "\t<image file='.\\depth_color\\col_depth" + std::to_string(j) + ".png'>" << std::endl;
			fout << "\t\t<box top='" + std::to_string(rects[j][0]) + "' left='" + std::to_string(rects[j][1]) + "' width='" + std::to_string(rects[j][2]) + "' height='" + std::to_string(rects[j][3]) + "'/>" << std::endl;
			fout << "\t</image>" << std::endl;
		}
	}
	fout << "</images>" << std::endl;
	fout << "</dataset>" << std::endl;
	fout.close();
}

void train(std::string filepath_base) {
	std::cout << "Training HOG detector" << std::endl;
	dlib::array<dlib::array2d<unsigned char>> faces;
	std::vector<std::vector<dlib::rectangle>> ROIs;
	std::cout << "Initialized" << std::endl;
	dlib::load_image_dataset(faces, ROIs, filepath_base + "\\depth_stream_training_small.xml");
	std::cout << "Loaded data" << std::endl;
	dlib::add_image_left_right_flips(faces, ROIs);
	std::cout << "Flipped data" << std::endl;
	dlib::scan_fhog_pyramid<dlib::pyramid_down<6>> scanner;
	scanner.set_detection_window_size(150, 150);
	scanner.set_nuclear_norm_regularization_strength(1.0);
	dlib::structural_object_detection_trainer<dlib::scan_fhog_pyramid<dlib::pyramid_down<6>>> HOG_trainer(scanner);
	HOG_trainer.set_num_threads(4);
	HOG_trainer.set_c(1);
	HOG_trainer.be_verbose();
	HOG_trainer.set_epsilon(0.01);
	std::cout << "Parameters set, starting training" << std::endl;
	dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6>>> detector = HOG_trainer.train(faces, ROIs);
	std::cout << "training results: " << dlib::test_object_detection_function(detector, faces, ROIs) << std::endl;
	std::cout << "num filters: " << num_separable_filters(detector) << std::endl;
	dlib::image_window gradients(draw_fhog(detector), "Gradient histogram");
	dlib::serialize("depth_detector.svm") << detector;
	std::cout << "Press enter" << std::endl;
	std::cin.get();
}

void multitrain(std::string filepath_base) {
	std::cout << "Training multiple HOG detectors" << std::endl;
	dlib::array<dlib::array2d<unsigned char>> faces;
	std::vector<std::vector<dlib::rectangle>> ROIs;
	dlib::load_image_dataset(faces, ROIs, filepath_base + "\\depth_stream_training_small.xml");
	dlib::array<dlib::array2d<unsigned char>> test_faces;
	std::vector<std::vector<dlib::rectangle>> test_ROIs;
	dlib::load_image_dataset(test_faces, test_ROIs, filepath_base + "\\depth_stream_testing.xml");
	dlib::add_image_left_right_flips(faces, ROIs);
	dlib::scan_fhog_pyramid<dlib::pyramid_down<6>> scanner;
	scanner.set_detection_window_size(150, 150);
	scanner.set_nuclear_norm_regularization_strength(1.0);
	std::vector<dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6>>>> detectors;
	std::vector<dlib::matrix<double, 1, 3>> results;
	dlib::structural_object_detection_trainer<dlib::scan_fhog_pyramid<dlib::pyramid_down<6>>> HOG_trainer(scanner);
	HOG_trainer.set_num_threads(8);
	HOG_trainer.be_verbose();
	HOG_trainer.set_epsilon(0.001);
	HOG_trainer.set_max_runtime(std::chrono::minutes(5));
	for (int i = -5; i <= 5; ++i) {
		std::cout << "Iteration " + std::to_string(i + 5) << std::endl;
		HOG_trainer.set_c(std::pow(10.0f, i));
		dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6>>> detector = HOG_trainer.train(faces, ROIs);
		results.push_back(dlib::test_object_detection_function(detector, test_faces, test_ROIs));
		detectors.push_back(detector);
	};
	for (int i = 0; i < 10; ++i) {
		std::cout << "Iteration " + std::to_string(i) + ", C=" + std::to_string(std::pow(10.0f, i - 5)) + ", results: " << results[i] << std::endl;
	}
	int choice = -1;
	while (choice < 0 || choice > 9) {
		std::cout << "Choose best detector" << std::endl;
		std::cin >> choice;
	}
	dlib::serialize("depth_detector.svm") << detectors[choice];
	std::cout << "Press enter" << std::endl;
	std::cin.get();
}

void test(std::string filepath_base) {
	std::cout << "Testing HOG detector" << std::endl;
	dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6>>> detector;
	dlib::deserialize("depth_detector.svm") >> detector;
	dlib::array<dlib::array2d<unsigned char>> faces;
	std::vector<std::vector<dlib::rectangle>> ROIs;
	dlib::load_image_dataset(faces, ROIs, filepath_base + "\\depth_stream_testing.xml");
	dlib::add_image_left_right_flips(faces, ROIs);
	std::cout << "testing results:  " << dlib::test_object_detection_function(detector, faces, ROIs) << std::endl;
	std::cout << "Press enter" << std::endl;
	std::cin.get();
}

void test_int(std::string filepath_base) {
	std::cout << "Interactive test" << std::endl;
	dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6>>> detector;
	dlib::deserialize("depth_detector.svm") >> detector;
	dlib::array<dlib::array2d<unsigned char>> faces;
	std::vector<std::vector<dlib::rectangle>> ROIs;
	dlib::load_image_dataset(faces, ROIs, filepath_base + "\\depth_stream_testing.xml");
	//dlib::add_image_left_right_flips(faces, ROIs);
	dlib::image_window win;
	for (unsigned long i = 0; i < faces.size(); ++i)
	{
		std::vector<dlib::rectangle> dets = detector(faces[i]);
		win.clear_overlay();
		win.set_image(faces[i]);
		win.add_overlay(dets, dlib::rgb_pixel(255, 0, 0));
		std::cout << "Hit enter to process the next image..." << std::endl;
		std::cin.get();
	}
}

void debug(std::string filepath_base) {
	dlib::array<dlib::array2d<unsigned char>> faces;
	std::vector<std::vector<dlib::rectangle>> ROIs;
	dlib::load_image_dataset(faces, ROIs, filepath_base + "\\depth_stream_testing.xml");
	dlib::image_window win;
	for (unsigned long i = 0; i < faces.size(); ++i)
	{
		win.clear_overlay();
		win.set_image(faces[i]);
		win.add_overlay(ROIs[i][0], dlib::rgb_pixel(255, 0, 0));
		std::cout << "Hit enter to process the next image..." << std::endl;
		std::cin.get();
	}
}

int main(int argc, char * argv[]) try{
	std::string filepath_base = argv[2];
	if (strcmp(argv[1], "build-test") == 0 || strcmp(argv[1], "build-train") == 0) {
		build(filepath_base, strcmp(argv[1], "build-test") == 0);
	}
	if (strcmp(argv[1], "build-small") == 0) {
		build_small2(filepath_base);
	}
	if (strcmp(argv[1], "train") == 0) {
		train(filepath_base);
	}
	if (strcmp(argv[1], "multitrain") == 0) {
		multitrain(filepath_base);
	}
	if (strcmp(argv[1], "test") == 0) {
		test(filepath_base);
	}
	if (strcmp(argv[1], "test-int") == 0) {
		test_int(filepath_base);
	}
	if (strcmp(argv[1], "full-train") == 0) {
		build(filepath_base, true);
		build(filepath_base, false);
		train(filepath_base);
		test(filepath_base);
	}
	if (strcmp(argv[1], "debug") == 0) {
		debug(filepath_base);
	}
	std::cout << "FINISHED" << std::endl;
}
catch (dlib::error e) {
	std::cout << e.what() << std::endl;
}




