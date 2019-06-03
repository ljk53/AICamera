#pragma once

// @generated by aten/src/ATen/gen.py

#include <ATen/CPUTypeDefault.h>
#include <ATen/Context.h>
#include <ATen/CheckGenerator.h>



#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

struct QuantizedCPUType final : public CPUTypeDefault {
  explicit QuantizedCPUType();
  virtual Backend backend() const override;
  virtual const char * toString() const override;
  virtual TypeID ID() const override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) override;
  Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const override;
  Tensor _empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point) const override;
  Tensor dequantize(const Tensor & self) const override;
  Scalar q_scale(const Tensor & self) const override;
  Scalar q_zero_point(const Tensor & self) const override;
  Tensor int_repr(const Tensor & self) const override;
  Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) const override;

 private:
  ScalarType infer_scalar_type(const Tensor & t) const {
    return t.scalar_type();
  }
  ScalarType infer_scalar_type(const TensorList & tl) const {
    TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
    return tl[0].scalar_type();
  }
};

} // namespace at
