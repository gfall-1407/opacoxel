from . import _C  
  
def hello_world():  
    """Simple hello world function that prints from C++."""  
    return _C.hello_world()