#pragma once
#include <ATen/Backend.h>


namespace at {

template <typename FnPtr>
inline void register_extension_backend_op(
    Backend backend,
    const char * schema,
    FnPtr fn) {
      switch (backend) {

        default:
          AT_ERROR("Invalid extension backend: ", toString(backend));
    }
}

} // namespace at
