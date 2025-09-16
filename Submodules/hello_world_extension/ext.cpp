#include <torch/extension.h>  
#include <iostream>  

#include <torch/extension.h>
#include <stdio.h>

void hello_world_extension() {  
    printf("Hello World\n");  
}  

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  
    m.def("hello_world", &hello_world_extension);  
}