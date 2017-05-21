//
//  Swapaxes.swift
//  MPSCNNfeeder
//
//  Created by Kazu Komoto on 12/17/16.
//  Copyright © 2016 Kazu Komoto. All rights reserved.
//
/*
 The MIT License (MIT)
 
 Copyright (c) 2016 Kazu Komoto
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */
/* 
   Helper class to feed the original array to Metal Performance Shaders.
   Changing from order of weights [kH kW iC oC] to MPS accepted order of weights i.e. [oC kH kW iC]
   reference: https://forums.developer.apple.com/message/195288#195288
*/

import Foundation

class SwapAxes {
  
  // swap two axes for 1d (flat) array that was originally 2d tensor
  static func for2dFlatArray<T>(originalArray: [T], axis1: Int, axis2: Int, dimensionOfArray: inout [Int]) -> [T] {
    assert(axis1 != axis2)
    assert(dimensionOfArray.count == 2)
    assert(axis1 < dimensionOfArray.count)
    assert(axis2 < dimensionOfArray.count)
    assert(originalArray.count == dimensionOfArray.reduce(1, *))
    assert((T.self == Float.self) || (T.self == Double.self))
    
    var newArray = Array<T>()
    
    for j in 0..<dimensionOfArray[1] {
      for i in 0..<dimensionOfArray[0] {
        newArray.append(originalArray[i*dimensionOfArray[1] + j])
      }
    }
    dimensionOfArray = [dimensionOfArray[1], dimensionOfArray[0]]
    return newArray
  }
  
  
  // swap two axes for 1d (flat) array that was originally 4d tensor
  static func for4dFlatArray<T>(originalArray: [T], axis1: Int, axis2: Int, dimensionOfArray: inout [Int]) -> [T] {
    assert(axis1 != axis2)
    assert(dimensionOfArray.count == 4)
    assert(axis1 < dimensionOfArray.count)
    assert(axis2 < dimensionOfArray.count)
    assert(originalArray.count == dimensionOfArray.reduce(1, *))
    assert((T.self == Float.self) || (T.self == Double.self))
    
    var newArray = Array<T>()
    
    if ((axis1 == 0 && axis2 == 1) || (axis1 == 1 && axis2 == 0)) {
      
      for j in 0..<dimensionOfArray[1] {
        for i in 0..<dimensionOfArray[0] {
          for k in 0..<dimensionOfArray[2] {
            for l in 0..<dimensionOfArray[3] {
              newArray.append(originalArray[i*dimensionOfArray[1]*dimensionOfArray[2]*dimensionOfArray[3] + j*dimensionOfArray[2]*dimensionOfArray[3] + k*dimensionOfArray[3] + l])
            }
          }
        }
      }
      dimensionOfArray = [dimensionOfArray[1], dimensionOfArray[0], dimensionOfArray[2], dimensionOfArray[3]]

    } else if ((axis1 == 0 && axis2 == 2) || (axis1 == 2 && axis2 == 0)) {
      
      for k in 0..<dimensionOfArray[2] {
        for j in 0..<dimensionOfArray[1] {
          for i in 0..<dimensionOfArray[0] {
            for l in 0..<dimensionOfArray[3] {
              newArray.append(originalArray[i*dimensionOfArray[1]*dimensionOfArray[2]*dimensionOfArray[3] + j*dimensionOfArray[2]*dimensionOfArray[3] + k*dimensionOfArray[3] + l])
            }
          }
        }
      }
      dimensionOfArray = [dimensionOfArray[2], dimensionOfArray[1], dimensionOfArray[0], dimensionOfArray[3]]
      
    } else if ((axis1 == 0 && axis2 == 3) || (axis1 == 3 && axis2 == 0)) {
      
      for l in 0..<dimensionOfArray[3] {
        for j in 0..<dimensionOfArray[1] {
          for k in 0..<dimensionOfArray[2] {
            for i in 0..<dimensionOfArray[0] {
              newArray.append(originalArray[i*dimensionOfArray[1]*dimensionOfArray[2]*dimensionOfArray[3] + j*dimensionOfArray[2]*dimensionOfArray[3] + k*dimensionOfArray[3] + l])
            }
          }
        }
      }
      dimensionOfArray = [dimensionOfArray[3], dimensionOfArray[1], dimensionOfArray[2], dimensionOfArray[0]]
      
    } else if ((axis1 == 1 && axis2 == 2) || (axis1 == 2 && axis2 == 1)) {
      
      for i in 0..<dimensionOfArray[0] {
        for k in 0..<dimensionOfArray[2] {
          for j in 0..<dimensionOfArray[1] {
            for l in 0..<dimensionOfArray[3] {
              newArray.append(originalArray[i*dimensionOfArray[1]*dimensionOfArray[2]*dimensionOfArray[3] + j*dimensionOfArray[2]*dimensionOfArray[3] + k*dimensionOfArray[3] + l])
            }
          }
        }
      }
      dimensionOfArray = [dimensionOfArray[0], dimensionOfArray[2], dimensionOfArray[1], dimensionOfArray[3]]
      
    } else if ((axis1 == 1 && axis2 == 3) || (axis1 == 3 && axis2 == 1)) {
      
      for i in 0..<dimensionOfArray[0] {
        for l in 0..<dimensionOfArray[3] {
          for k in 0..<dimensionOfArray[2] {
            for j in 0..<dimensionOfArray[1] {
              newArray.append(originalArray[i*dimensionOfArray[1]*dimensionOfArray[2]*dimensionOfArray[3] + j*dimensionOfArray[2]*dimensionOfArray[3] + k*dimensionOfArray[3] + l])
            }
          }
        }
      }
      dimensionOfArray = [dimensionOfArray[0], dimensionOfArray[3], dimensionOfArray[2], dimensionOfArray[1]]
      
    } else if ((axis1 == 2 && axis2 == 3) || (axis1 == 3 && axis2 == 2)) {
      
      for i in 0..<dimensionOfArray[0] {
        for j in 0..<dimensionOfArray[1] {
          for l in 0..<dimensionOfArray[3] {
            for k in 0..<dimensionOfArray[2] {
              newArray.append(originalArray[i*dimensionOfArray[1]*dimensionOfArray[2]*dimensionOfArray[3] + j*dimensionOfArray[2]*dimensionOfArray[3] + k*dimensionOfArray[3] + l])
            }
          }
        }
      }
      dimensionOfArray = [dimensionOfArray[0], dimensionOfArray[1], dimensionOfArray[3], dimensionOfArray[2]]

      
    } else {
      fatalError("Didn't match all the case")
    }
    
    return newArray
  }
 
}
