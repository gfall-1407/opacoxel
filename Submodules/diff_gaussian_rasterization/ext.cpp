#include <torch/extension.h>
#include <iostream>

void HelloWorld() {
  std::cout << "Hello, World!" << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hello_world", &HelloWorld);
}