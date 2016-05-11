#include <vector>

#include "caffe/layers/swap_dim_layer.hpp"


namespace caffe {

template <typename Dtype>
__global__ void Forward(const int nthreads,
	const Dtype* bottom_data, const int num, int channels, int height, int width,
	Dtype *top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;

		int bottom_index =  w * height * channels * num +  h * channels * num + c * num + n;
		top_data[index] = bottom_data[bottom_index]; 
	}
}


template <typename Dtype>
void SwapDimLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	const Dtype *bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	const int count = top[0]->count();
	const int num_ = top[0]->num();
	const int channels_ = top[0]->channels();
	const int height_ = top[0]->height();
	const int width_ = top[0]->width();
 
	Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, num_, channels_, height_, width_, top_data)
	
	CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void Backward(const int nthreads,
	const Dtype* top_diff, const int num, int channels, int height, int width,
	Dtype *bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;

		int bottom_index =  w * height * channels * num +  h * channels * num + c * num + n;
		bottom_diff[bottom_index] = top_diff[index]; 
	}
}


template <typename Dtype>
void SwapDimLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}

	const Dtype *top_diff = top[0]->gpu_diff();
	Dtype* bottom_Diff = bottom[0]->mutable_gpu_diff();
	const int count = top[0]->count();
	const int num_ = top[0]->num();
	const int channels = top[0]->channels();
	const int height_ = top[0]->height();
	const int width_ = top[0]->width();
	caffe_gpu_set(count, Dtype(0.), bottom_diff);


	Backward<Dtype><<<CAFFE_GET_BLOCKS(count)>>>, CAFFE_CUDA_NUM_THREADS>>>(
		count, top_diff, num_, channels_, height_, width_, bottom_diff);
	
	CUDA_POST_KERNEL_CHECK;	
}

INSTANTIATE_LAYER_GPU_FUNCS(SwapDimLayer);
} // namespace caffe
