from . import _C

def hello_world():
    # C++ module defines the Python binding as "hello_world"
    # (m.def("hello_world", &hello_world_extension)), so call that.
    return _C.hello_world()