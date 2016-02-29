#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PoolingBatchLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PoolingBatchLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 5, 1, 1);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PoolingBatchLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward5() {
  // Test for 2 x 5 x 1 x 1 input
    LayerParameter layer_param;
    PoolingParameter *pooling_param = layer_param.mutable_pooling_param();
    pooling_param -> set_kernel_h(1);
    pooling_param -> set_kernel_w(1);
    const int num = 2;
    const int channels = 5;
    blob_bottom_ -> Reshape(num, channels, 1, 1);
    //Input: [1 2 3 4 5]
    //       [4 1 2 6 2]
    vector<int> data; data.clear();
    data.push_back(1);
    data.push_back(2);
    data.push_back(3);
    data.push_back(4);
    data.push_back(5);
    data.push_back(4);
    data.push_back(1);
    data.push_back(2);
    data.push_back(6);
    data.push_back(2);
    for(int i = 0; i < 10; i++)
      blob_bottom_ -> mutable_cpu_data()[i] = data[i];

    PoolingBatchLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), 1);
    EXPECT_EQ(blob_top_->channels(), 1);
    EXPECT_EQ(blob_top_->height(), channels);
    EXPECT_EQ(blob_top_->width(), 1);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), 1);
      EXPECT_EQ(blob_top_mask_->channels(), 1);
      EXPECT_EQ(blob_top_mask_->height(), channels);
      EXPECT_EQ(blob_top_mask_->width(), 1);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output (1 x 1 x 5 x 1)
    //     [4 2 3 6 5]
    data.clear();
    data.push_back(4);
    data.push_back(2);
    data.push_back(3);
    data.push_back(6);
    data.push_back(5);

    for(int i = 0; i < 5; i++)
      EXPECT_EQ(blob_top_->cpu_data()[i], data[i]);

    if (blob_top_vec_.size() > 1) {
      for(int i = 0; i < 5; i++)
        EXPECT_EQ(blob_top_mask_->cpu_data()[i],  data[i]);      
    }
  }
};

TYPED_TEST_CASE(PoolingBatchLayerTest, TestDtypesAndDevices);

TYPED_TEST(PoolingBatchLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(1);
  pooling_param->set_stride(1);
  PoolingBatchLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->width(), 1);
}


TYPED_TEST(PoolingBatchLayerTest, TestForwardMax) {
  this->TestForward5();
}

TYPED_TEST(PoolingBatchLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 1; kernel_h <= 1; kernel_h++) {
    for (int kernel_w = 1; kernel_w <= 1; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      PoolingBatchLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}


// TYPED_TEST(PoolingBatchLayerTest, TestGradientMaxTopMask) {
//   typedef typename TypeParam::Dtype Dtype;
//   for (int kernel_h = 1; kernel_h <= 1; kernel_h++) {
//     for (int kernel_w = 1; kernel_w <= 1; kernel_w++) {
//       LayerParameter layer_param;
//       PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
//       pooling_param->set_kernel_h(kernel_h);
//       pooling_param->set_kernel_w(kernel_w);
//       pooling_param->set_stride(1);
//       pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
//       this->blob_top_vec_.push_back(this->blob_top_mask_);
//       PoolingBatchLayer<Dtype> layer(layer_param);
//       GradientChecker<Dtype> checker(1e-4, 1e-2);
//       checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//           this->blob_top_vec_);
//       this->blob_top_vec_.pop_back();
//     }
//   }
// }

}  // namespace caffe
