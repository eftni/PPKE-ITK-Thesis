void kernel gradient(global const ushort* input, global ushort* x_output,  global uint* y_output,  unsigned int width,  unsigned int height){
	const int thread = get_global_id(0);
	for(unsigned int i = 0; i < width; ++i){
		if(i != 0 && i != width-1) x_output[thread*width + i] = input[thread*width + i+1] - input[thread*width + i-1];
		if(thread != 0 && thread != height-1) y_output[thread*width + i] = input[(thread-1)*width + i] - input[(thread+1)*width + i];
	}
};