//
//  SlimMPSCNN.swift
//  HandGestureCNN
//
//  Created by 杨萧玉 on 2017/5/21.
//  Copyright © 2017年 杨萧玉. All rights reserved.
//
/*
 Abstract:
 This file describes slimmer routines to create some common MPSCNNFunctions,
 it is useful especially to fetch network parameters from .dat files or HDF5 file.
 This is based on SlimMPSCNN.swift provided by Apple.
 */

import Foundation
import MetalPerformanceShaders
import HDF5Kit

/**
 This depends on MetalPerformanceShaders.framework
 
 The SlimMPSCNNConvolution is a wrapper class around MPSCNNConvolution used to encapsulate:
 - making an MPSCNNConvolutionDescriptor,
 - adding network parameters (weights and bias binaries by memory mapping the binaries)
 - getting our convolution layer
 */

private let h5FileName = "fc_model"

class SlimMPSCNNConvolution: MPSCNNConvolution{
    /**
     A property to keep info from init time whether we will pad input image or not for use during encode call
     */
    private var padding = true
    
    /**
     Initializes a fully connected kernel.
     
     - Parameters:
     - kernelWidth: Kernel Width
     - kernelHeight: Kernel Height
     - inputFeatureChannels: Number feature channels in input of this layer
     - outputFeatureChannels: Number feature channels from output of this layer
     - neuronFilter: A neuronFilter to add at the end as activation, default is nil
     - device: The MTLDevice on which this SlimMPSCNNConvolution filter will be used
     - kernelParamsBinaryName: name of the layer to fetch kernelParameters by adding a prefix "weights_" or "bias_"
     - padding: Bool value whether to use padding or not
     - strideXY: Stride of the filter
     - destinationFeatureChannelOffset: FeatureChannel no. in the destination MPSImage to start writing from, helps with concat operations
     - groupNum: if grouping is used, default value is 1 meaning no groups
     
     - Returns:
     A valid SlimMPSCNNConvolution object or nil, if failure.
     */
    
    
    init(kernelWidth: UInt, kernelHeight: UInt, inputFeatureChannels: UInt, outputFeatureChannels: UInt, neuronFilter: MPSCNNNeuron? = nil, device: MTLDevice, kernelParamsBinaryName: String, padding willPad: Bool = true, strideXY: (UInt, UInt) = (1, 1), destinationFeatureChannelOffset: UInt = 0, groupNum: UInt = 1){
        
        // check whether dat file exists in Bundle or Temporary directory in the client device
        let fileManager = FileManager.default
        var wtPath: String?
        var bsPath: String?
        
        let wtPathBundle = Bundle.main.path(forResource: "weights_" + kernelParamsBinaryName, ofType: "dat")
        let bsPathBundle = Bundle.main.path(forResource: "bias_" + kernelParamsBinaryName, ofType: "dat")
        
        // Note: Switch from Temporary directory to Library/Caches/ if needed to persist the data after closing)
        let tmpPath = URL(fileURLWithPath: NSTemporaryDirectory())
        let wtPathTmp = tmpPath.appendingPathComponent("weights_" + kernelParamsBinaryName).appendingPathExtension("dat")
        let bsPathTmp = tmpPath.appendingPathComponent("bias_" + kernelParamsBinaryName).appendingPathExtension("dat")
        
        if (wtPathBundle != nil) && (bsPathBundle != nil) {
            
            // Use parameters in Bundle
            wtPath = wtPathBundle
            bsPath = bsPathBundle
            
        } else {
            if fileManager.fileExists(atPath:wtPathTmp.path) && fileManager.fileExists(atPath:bsPathTmp.path) {
                extractHDF5(h5Name: h5FileName)
            }
            // Use parameters in Tmp
            wtPath = wtPathTmp.path
            bsPath = bsPathTmp.path
        }
        
        // calculate the size of weights and bias required to be memory mapped into memory
        let sizeBias = outputFeatureChannels * UInt(MemoryLayout<Float>.size)
        let sizeWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannels * UInt(MemoryLayout<Float>.size)
        
        
        // open file descriptors in read-only mode to parameter files
        let fd_w = open( wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        let fd_b = open( bsPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        
        assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")
        assert(fd_b != -1, "Error: failed to open output file at \""+bsPath!+"\"  errno = \(errno)\n")
        
        // memory map the parameters
        let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0)
        let hdrB = mmap(nil, Int(sizeBias), PROT_READ, MAP_FILE | MAP_SHARED, fd_b, 0)
        
        // cast Void pointers to Float
        let w = UnsafePointer<Float>(hdrW!.assumingMemoryBound(to: Float.self))
        let b = UnsafePointer<Float>(hdrB!.assumingMemoryBound(to: Float.self))
        
        assert(w != UnsafePointer<Float>(bitPattern: -1), "mmap failed with errno = \(errno)")
        assert(b != UnsafePointer<Float>(bitPattern: -1), "mmap failed with errno = \(errno)")
        
        // create appropriate convolution descriptor with appropriate stride
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelWidth),
                                                   kernelHeight: Int(kernelHeight),
                                                   inputFeatureChannels: Int(inputFeatureChannels),
                                                   outputFeatureChannels: Int(outputFeatureChannels),
                                                   neuronFilter: neuronFilter)
        convDesc.strideInPixelsX = Int(strideXY.0)
        convDesc.strideInPixelsY = Int(strideXY.1)
        
        assert(groupNum > 0, "Group size can't be less than 1")
        convDesc.groups = Int(groupNum)
        
        // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
        super.init(device: device,
                   convolutionDescriptor: convDesc,
                   kernelWeights: w,
                   biasTerms: b,
                   flags: MPSCNNConvolutionFlags.none)
        self.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        
        // set padding for calculation of offset during encode call
        padding = willPad
        
        // unmap files at initialization of MPSCNNConvolution, the weights are copied and packed internally we no longer require these
        assert(munmap(hdrW, Int(sizeWeights)) == 0, "munmap failed with errno = \(errno)")
        assert(munmap(hdrB, Int(sizeBias))    == 0, "munmap failed with errno = \(errno)")
        
        // close file descriptors
        close(fd_w)
        close(fd_b)
        
    }
    
    /**
     Encode a MPSCNNKernel into a command Buffer. The operation shall proceed out-of-place.
     
     We calculate the appropriate offset as per how TensorFlow calculates its padding using input image size and stride here.
     
     This [Link](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn.py) has an explanation in header comments how tensorFlow pads its convolution input images.
     
     - Parameters:
     - commandBuffer: A valid MTLCommandBuffer to receive the encoded filter
     - sourceImage: A valid MPSImage object containing the source image.
     - destinationImage: A valid MPSImage to be overwritten by result image. destinationImage may not alias sourceImage
     */
    override func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        // select offset according to padding being used or not
        if padding {
            let pad_along_height = ((destinationImage.height - 1) * strideInPixelsY + kernelHeight - sourceImage.height)
            let pad_along_width  = ((destinationImage.width - 1) * strideInPixelsX + kernelWidth - sourceImage.width)
            let pad_top = Int(pad_along_height / 2)
            let pad_left = Int(pad_along_width / 2)
            
            self.offset = MPSOffset(x: ((Int(kernelWidth)/2) - pad_left), y: (Int(kernelHeight/2) - pad_top), z: 0)
        }
        else{
            self.offset = MPSOffset(x: Int(kernelWidth)/2, y: Int(kernelHeight)/2, z: 0)
        }
        
        super.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
}

/**
 This depends on MetalPerformanceShaders.framework
 
 The SlimMPSCNNFullyConnected is a wrapper class around MPSCNNFullyConnected used to encapsulate:
 - making an MPSCNNConvolutionDescriptor,
 - adding network parameters (weights and bias binaries by memory mapping the binaries)
 - getting our fullyConnected layer
 */
class SlimMPSCNNFullyConnected: MPSCNNFullyConnected{
    /**
     Initializes a fully connected kernel.
     
     - Parameters:
     - kernelWidth: Kernel Width
     - kernelHeight: Kernel Height
     - inputFeatureChannels: Number feature channels in input of this layer
     - outputFeatureChannels: Number feature channels from output of this layer
     - neuronFilter: A neuronFilter to add at the end as activation, default is nil
     - device: The MTLDevice on which this SlimMPSCNNConvolution filter will be used
     - kernelParamsBinaryName: name of the layer to fetch kernelParameters by adding a prefix "weights_" or "bias_"
     - destinationFeatureChannelOffset: FeatureChannel no. in the destination MPSImage to start writing from, helps with concat operations
     
     - Returns:
     A valid SlimMPSCNNFullyConnected object or nil, if failure.
     */
    
    init(kernelWidth: UInt, kernelHeight: UInt, inputFeatureChannels: UInt, outputFeatureChannels: UInt, neuronFilter: MPSCNNNeuron? = nil, device: MTLDevice, kernelParamsBinaryName: String, destinationFeatureChannelOffset: UInt = 0){
        
        // check whether dat file exists in Bundle or Temporary directory in the client device
        let fileManager = FileManager.default
        var wtPath: String?
        var bsPath: String?
        
        let wtPathBundle = Bundle.main.path(forResource: "weights_" + kernelParamsBinaryName, ofType: "dat")
        let bsPathBundle = Bundle.main.path(forResource: "bias_" + kernelParamsBinaryName, ofType: "dat")
        
        // Note: Switch from Temporary directory to Library/Caches/ if needed to persist the data after closing)
        let tmpPath = URL(fileURLWithPath: NSTemporaryDirectory())
        let wtPathTmp = tmpPath.appendingPathComponent("weights_" + kernelParamsBinaryName).appendingPathExtension("dat")
        let bsPathTmp = tmpPath.appendingPathComponent("bias_" + kernelParamsBinaryName).appendingPathExtension("dat")
        
        if (wtPathBundle != nil) && (bsPathBundle != nil) {
            
            // Use parameters in Bundle
            wtPath = wtPathBundle
            bsPath = bsPathBundle
            
        } else {
            if !(fileManager.fileExists(atPath:wtPathTmp.path) && fileManager.fileExists(atPath:bsPathTmp.path)) {
                extractHDF5(h5Name: h5FileName)
            }
            // Use parameters in Tmp
            wtPath = wtPathTmp.path
            bsPath = bsPathTmp.path
        }
        
        
        // calculate the size of weights and bias required to be memory mapped into memory
        let sizeBias = outputFeatureChannels * UInt(MemoryLayout<Float>.size)
        let sizeWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannels * UInt(MemoryLayout<Float>.size)
        
        // open file descriptors in read-only mode to parameter files
        let fd_w = open(wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        let fd_b = open(bsPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        
        assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")
        assert(fd_b != -1, "Error: failed to open output file at \""+bsPath!+"\"  errno = \(errno)\n")
        
        // memory map the parameters
        let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0)
        let hdrB = mmap(nil, Int(sizeBias), PROT_READ, MAP_FILE | MAP_SHARED, fd_b, 0)
        
        // cast Void pointers to Float
        let w = UnsafePointer<Float>(hdrW!.assumingMemoryBound(to: Float.self))
        let b = UnsafePointer<Float>(hdrB!.assumingMemoryBound(to: Float.self))
        
        assert(w != UnsafePointer<Float>(bitPattern: -1), "mmap failed with errno = \(errno)")
        assert(b != UnsafePointer<Float>(bitPattern: -1), "mmap failed with errno = \(errno)")
        
        // create appropriate convolution descriptor (in fully connected, stride is always 1)
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelWidth),
                                                   kernelHeight: Int(kernelHeight),
                                                   inputFeatureChannels: Int(inputFeatureChannels),
                                                   outputFeatureChannels: Int(outputFeatureChannels),
                                                   neuronFilter: neuronFilter)
        
        // initialize the convolution layer by calling the parent's (MPSCNNFullyConnected's) initializer
        super.init(device: device,
                   convolutionDescriptor: convDesc,
                   kernelWeights: w,
                   biasTerms: b,
                   flags: MPSCNNConvolutionFlags.none)
        
        self.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        
        // unmap files at initialization of MPSCNNFullyConnected, the weights are copied and packed internally we no longer require these
        assert(munmap(hdrW, Int(sizeWeights)) == 0, "munmap failed with errno = \(errno)")
        assert(munmap(hdrB, Int(sizeBias))    == 0, "munmap failed with errno = \(errno)")
        
        // close file descriptors
        close(fd_w)
        close(fd_b)
    }
}

// Read parameters from HDF5 file and store to dat file in Tmp directory
func extractHDF5(h5Name: String) {
    // MARK: Parse HDF5 file
    guard let path = Bundle.main.path(forResource: h5Name, ofType: "h5") else {
        fatalError("Failed to get a path")
    }
    guard let file = File.open(path, mode: .readOnly) else {
        fatalError("Failed to open file at \(path)")
    }
    
    guard let layerNamesStringAttribute = file.openStringAttribute("layer_names") else {
        fatalError("Failed to open attribute 'layer_names'")
    }
    guard let layerNames = try? layerNamesStringAttribute.read() else {
        fatalError("Failed to get layer names")
    }
    
    // count used for file name later
    var countOfConvLayer = 0
    var countOfFcLayer   = 0
    var partOfFileName = ""
    
    for layerName in layerNames {
        guard let layerGroup = file.openGroup(layerName) else {
            fatalError("Failed to open group of \(layerName)")
        }
        for objectName in layerGroup.objectNames() {
            
            // only the layer that has parameters remain
            guard let wtDataset = layerGroup.openFloatDataset(objectName + "/kernel:0") else {
                fatalError("Failed to open data set of \(objectName)/kernel:0")
            }
            guard let bsDataset = layerGroup.openFloatDataset(objectName + "/bias:0") else {
                fatalError("Failed to open data set of \(objectName)/bias:0")
            }
            
            var dimension = wtDataset.space.dims
            guard var wtArray = try? wtDataset.read() else {
                fatalError("Failed to read data set of \(objectName)/kernel:0")
            }
            guard var bsArray = try? bsDataset.read() else {
                fatalError("Failed to read data set of \(objectName)/bias:0")
            }
            
            let wtLength = wtArray.count
            let bsLength = bsArray.count
            
            if dimension.count == 4 {
                // weights for convolution layer
                wtArray = SwapAxes.for4dFlatArray(originalArray: wtArray, axis1: 2, axis2: 3, dimensionOfArray: &dimension)
                wtArray = SwapAxes.for4dFlatArray(originalArray: wtArray, axis1: 1, axis2: 2, dimensionOfArray: &dimension)
                wtArray = SwapAxes.for4dFlatArray(originalArray: wtArray, axis1: 0, axis2: 1, dimensionOfArray: &dimension)
                
                countOfConvLayer += 1
                partOfFileName = "conv" + String(countOfConvLayer)
                
            } else if dimension.count == 2 {
                // weights for fully connected layer
                wtArray = SwapAxes.for2dFlatArray(originalArray: wtArray, axis1: 0, axis2: 1, dimensionOfArray: &dimension)
                
                countOfFcLayer += 1
                partOfFileName = "fc" + String(countOfFcLayer)
                
            } else {
                fatalError("Dataset's dimension is neither 4 (convolution layer) nor 2 (fully connected layer)")
            }
            
            let wtData = NSData(bytes: &wtArray, length: wtLength * MemoryLayout<Float>.size)
            let bsData = NSData(bytes: &bsArray, length: bsLength * MemoryLayout<Float>.size)
            
            let filePath = URL(fileURLWithPath: NSTemporaryDirectory())
            let wtFilePath = filePath.appendingPathComponent("weights_" + partOfFileName).appendingPathExtension("dat")
            let bsFilePath = filePath.appendingPathComponent("bias_" + partOfFileName).appendingPathExtension("dat")
            do {
                try wtData.write(to: wtFilePath, options: .atomic)
            } catch {
                print(error)
            }
            do {
                try bsData.write(to: bsFilePath, options: .atomic)
            } catch {
                print(error)
            }
        }
    }
}
