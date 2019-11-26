#include "vector"
#include "helper.hpp"
#include <utility>
#include <fstream>
#include <math.h>


class Face_detector {
private:
	dlib::frontal_face_detector detector;
	rs2::pipeline input;
	rs2::pipeline_profile profile;
	rs2::colorizer color_map;
	rs2_stream align_to;
	const std::string window_name;
	const unsigned int size_x, size_y;
	std::ofstream ROI_file;
	
	int saved = 0;

	class Face_display {
	private:
		const unsigned int size_x, size_y;
		const std::string window_name;
	protected:
	public:
		Face_display(const unsigned int x, const unsigned int y, const std::string name);
		~Face_display();
		void show_face(cv::Mat& image);
	};

	std::vector<Face_display*> face_windows;

protected:
public:
	Face_detector(const unsigned int, const unsigned int, const std::string);
	~Face_detector();
	bool save = false;
	cv::Mat* curr_disp;
	void process_input();
};