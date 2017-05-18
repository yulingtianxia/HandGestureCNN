#The MIT License (MIT)
#
#Copyright (c) 2016 Kazu Komoto
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

import os
import numpy as np
import h5py


def main():
    weights_path = 'mnist_model_weights.h5'

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path, 'r')
    layer_names = [n for n in f]
    # layer_names = [n for n in f.attrs['layer_names']]  # If you want to extract in order

    for layer_name in layer_names:
        g = f[layer_name]
        weight_names = [n for n in g]
        # weight_names = [n for n in g.attrs['weight_names']]

        if len(weight_names):
            for weight_name in weight_names:
                # Shape
                shape = g[weight_name].shape
                # Data type
                typeof = g[weight_name].dtype
                # Dataset
                dataset = g[weight_name][()]
                #     dataset = f[name][()].flatten()

                # Swap for MPS convolution and fully connected layer's input
                if len(shape) == 4:  # convolution layer's weight
                    dataset = swapaxes_for_MPS_convolution(dataset)
                elif len(shape) == 2:  # fully connected layer's weight
                    dataset = swapaxes_for_MPS_fullyconnected(dataset)
                elif len(shape) == 1:  # bias
                    pass
                else:
                    print("Error: Unexpected shape of weights/bias")

                # Open binary .dat file
                output_file = weight_name[:-2] + '.dat'  # Cut off ':0'
                # Write data to .dat file
                dataset.astype(typeof).tofile(output_file)
                # Log
                print("Binary packed "+weight_name+" shape: "+str(shape)+ " type: " +str(typeof))

    f.close()
    print('Model loaded.')



# Save the parameters to dat file
## Swap numpy array to fit for Metal Performance Shaders.
## Changing from order of weights [kH kW iC oC] to MPS accepted order of weights i.e. [oC kH kW iC]
## ref: https://forums.developer.apple.com/message/195288#195288
def swapaxes_for_MPS_convolution(nparray):
    nparray = np.swapaxes(nparray, 2, 3)
    nparray = np.swapaxes(nparray, 1, 2)
    nparray = np.swapaxes(nparray, 0, 1)
    return nparray

def swapaxes_for_MPS_fullyconnected(nparray):
    nparray = np.swapaxes(nparray, 0, 1)
    return nparray



if __name__ == "__main__":
  main()
