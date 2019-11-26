void kernel normals(global const ushort* input, global float* normals, unsigned int width,  unsigned int height){
	const int thread = get_global_id(0);
	float dx = 0;
	float dy = 0;
	for(unsigned int i = 0; i < width; ++i){
		if(i != 0 && i != width-1) dx = (input[thread*width + i+1] - input[thread*width + i-1])/2.0;
		if(thread != 0 && thread != height-1) dy = (input[(thread-1)*width + i] - input[(thread+1)*width + i])/2.0;
		float magnitude = (pow(dx,2) + pow(dy,2) + 1.0);
		normals[thread*width + i*3] = (-1*dx)/magnitude;
		normals[thread*width + i*3 + 1] = (-1*dy)/magnitude;
		normals[thread*width + i*3 + 2] = -0.1/magnitude;
	}
};