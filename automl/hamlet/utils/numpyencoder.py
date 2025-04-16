import json
import numpy as np

# https://github.com/hmallen/numpyencoder/blob/f8199a61ccde25f829444a9df4b21bcb2d1de8f2/numpyencoder/numpyencoder.py
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        
        if isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        
        if isinstance(obj, (np.ndarray)):
            return obj.tolist()
    
        if isinstance(obj, (np.bool_)):
            return bool(obj)

        if isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)
