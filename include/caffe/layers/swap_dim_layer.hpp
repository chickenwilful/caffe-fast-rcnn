#ifndef CAFFE_SWAP_DIM_LAYER_HPP_
#define CAFFE_SWAP_DIM_LAYER_HPP_

#include <vector>

#include <caffe/blob.hpp>
#include <caffe/layer.hpp>
#include <caffe/proto/caffe.pb.h>

namespace caffe {

/**
 * @brief Change the input Blob from [NxCxHxW] to [WxHxCxN]
 * Note: this layer change the input values
 */
template <typename Dtype>
class SwapDimLayer : public Layer<Dtype> {
 public:
	explicit SwapDimLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
		
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	
};

} // namespace caffe

#endif // CAFFE_SWAP_DIM_LAYER_HPP_
