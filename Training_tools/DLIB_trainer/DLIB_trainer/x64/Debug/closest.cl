void kernel closest(global const ushort* input, global ushort* output,  global uint* indices,  unsigned int width,  unsigned int height){
	const int thread = get_global_id(0);
	ushort min = input[thread*height];
	uint min_index = thread*height;
	for(unsigned int i = 1; i < width; ++i){
		if(input[thread*height+i] < min && input[thread*height+i] != 0){
			min = input[thread*height+i];
			min_index = thread*height+i;
		}
	}
	output[thread] = min;
	indices[thread] = min_index;
};