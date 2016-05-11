#include <vector>
#include "caffe/layers/swap_dim_layer.hpp"

namespace caffe {

template <typename Dtype>
void SwapDimLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) { 
	// TODO: get params and check validity
}

template <typename Dtype>
void SwapDimLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
			const vector<Blob<Dtype>*>& top) {
	if (bottom[0]->num_axes() == 2)
		bottom[0] -> Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);

	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
		<< "corresponding to (num, channels, height, width)";
	
	top[0]->Reshape(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels(), bottom[0]->num());
}

template <typename Dtype>
void SwapDimLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {

	Dtype *top_data = top[0]->mutable_cpu_data();
	const Dtype *bottom_data = bottom[0]->cpu_data();

	int i = 0;
	for (int n = 0; n < bottom[0]->num(); ++n)
		for (int c = 0; c < bottom[0]->channels(); ++c)
			for (int h = 0; h < bottom[0]->height(); ++h)
				for (int w = 0; w < bottom[0]->width(); ++w) 
					top_data[top[0]->offset(w, h, c, n)] = bottom_data[i++];
}

template <typename Dtype>
void SwapDimLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}

	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	
	int i = 0;
	for (int n = 0; n < top[0]->num(); ++n)
		for (int c = 0; c < top[0]->channels(); ++c)
			for (int h = 0; h < top[0]->height(); ++h)
				for (int w = 0; w < top[0]->width(); ++w)
					bottom_diff[bottom[0]->offset(w, h, c, n)] = top_diff[i++];
	
	for (int n = 0; n < bottom[0]->num(); ++n)
		for (int c = 0; c < bottom[0]->channels(); ++c)
			for (int h = 0; h < bottom[0]->height(); ++h)
				for (int w = 0; w < bottom[0]->width(); ++w)
					CHECK_EQ(bottom[0]->diff_at(n, c, h, w), top[0]->diff_at(w, h, c, n));
}


#ifdef CPU_ONLY
STUB_GPU(SwapDimLayer);
#endif

INSTANTIATE_CLASS(SwapDimLayer);

} // namespace caffe
