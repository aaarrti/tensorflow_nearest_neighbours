#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <filesystem>
#include <sys/_types/_int32_t.h>
#include <dlfcn.h>

#include "tensorflow/c/kernels.h"
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <iostream>

@protocol TF_MetalStream

- (dispatch_queue_t)queue;
- (id<MTLCommandBuffer>) currentCommandBuffer;
- (void)commit;
- (void)commitAndWait;

@end

// The singleton class for kernel library.
class KernelLibrarySingleton {
   public:
    static KernelLibrarySingleton& getInstance() {
        if (sInstance == nullptr) {
            sInstance = new KernelLibrarySingleton();

            printf("Loading kernel library...\n");

            @autoreleasepool {
                // Finding the metallib path.
                NSString* libraryFile = @"nearest_neighbours_kernels.metallib";
                {
                    Dl_info info;
                    if (dladdr(reinterpret_cast<const void*>(&getInstance), &info) != 0) {
                        libraryFile = [NSString stringWithCString:info.dli_fname encoding:[NSString defaultCStringEncoding]];
                        libraryFile = [libraryFile stringByReplacingOccurrencesOfString:@".so" withString:@".metallib"];
                    }
                }
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();

                NSError* error = nil;
                NSURL *libraryUrl = [NSURL URLWithString:libraryFile];
                library = [device newLibraryWithURL:libraryUrl error:&error];

                if (!library) {
                    printf("Compilation error: %s\n", [[error description] UTF8String]);
                    abort();
                }
            }
        }
        return *sInstance;
    }

   public:
    static id<MTLLibrary> library;

   private:
    KernelLibrarySingleton() {}
    static KernelLibrarySingleton* sInstance;
};

KernelLibrarySingleton* KernelLibrarySingleton::sInstance = nullptr;
id<MTLLibrary> KernelLibrarySingleton::library = nil;



typedef struct NearestNeighboursOp {} MetalNearestNeighboursOp;


static void* MetalNearestNeighboursOp_Create(TF_OpKernelConstruction* ctx) {
    auto* kernel = new MetalNearestNeighboursOp;
    return kernel;
}

static void MetalNearestNeighboursOp_Delete(void* kernel) {
    delete static_cast<MetalNearestNeighboursOp *>(kernel);
}

static void MetalNearestNeighboursOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
    auto* k = static_cast<MetalNearestNeighboursOp *>(kernel);

    /*

    TF_Status* status = TF_NewStatus();

    TF_Tensor* inputs = nullptr;
    TF_GetInput(ctx, 0, &inputs, status);

    TF_Tensor* embeddings = nullptr;
    TF_GetInput(ctx, 1, &embeddings, status);

    TF_Tensor* hashmap_offsets = nullptr;
    TF_GetInput(ctx, 2, &hashmap_offsets, status);

    TF_DataType dataType = TF_TensorType(embeddings);



    std::vector<int64_t> inputs_shape = getShape(inputs);
    std::vector<int64_t> embeddings_shape = getShape(embeddings);
    std::vector<int64_t> offsets_shape = getShape(hashmap_offsets);

    int32_t B = inputs_shape[0];
    int32_t D = inputs_shape[1];
    int32_t L = offsets_shape[0] - 1;
    int32_t C = embeddings_shape[1];

    std::vector<int64_t> output_shape{inputs_shape[0],
                                      embeddings_shape[embeddings_shape.size() - 1] * L};

    TF_Tensor* outputs = TF_AllocateOutput(ctx, 0, dataType, (int64_t*)output_shape.data(),
                                           output_shape.size(), 0, status);

    if (TF_GetCode(status) != TF_OK) {
        printf("allocation failed: %s\n", TF_Message(status));
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteTensor(inputs);
        TF_DeleteTensor(embeddings);
        TF_DeleteTensor(hashmap_offsets);
        TF_DeleteTensor(outputs);
        TF_DeleteStatus(status);
        return;
    }

    @autoreleasepool {

        id<TF_MetalStream> metalStream = (id<TF_MetalStream>)(TF_GetStream(ctx, status));

        if (TF_GetCode(status) != TF_OK) {
            printf("no stream was found: %s\n", TF_Message(status));
            TF_OpKernelContext_Failure(ctx, status);
            TF_DeleteTensor(inputs);
            TF_DeleteTensor(embeddings);
            TF_DeleteTensor(hashmap_offsets);
            TF_DeleteTensor(outputs);
            TF_DeleteStatus(status);
            return;
        }

        dispatch_sync(metalStream.queue, ^() {
          @autoreleasepool {
              id<MTLCommandBuffer> commandBuffer = metalStream.currentCommandBuffer;
              id<MTLDevice> device = commandBuffer.device;

              NSError* error = nil;
              id<MTLLibrary> library = KernelLibrarySingleton::getInstance().library;

              id<MTLFunction> function = nil;

              function = [[library newFunctionWithName:@"NearestNeighbours"] autorelease];

              id<MTLComputePipelineState> pipeline =
                  [device newComputePipelineStateWithFunction:function error:&error];
              assert(pipeline);

              id<MTLBuffer> inputsBuffer = (id<MTLBuffer>)TF_TensorData(inputs);
              id<MTLBuffer> embeddingsBuffer = (id<MTLBuffer>)TF_TensorData(embeddings);
              id<MTLBuffer> offsetsBuffer = (id<MTLBuffer>)TF_TensorData(hashmap_offsets);
              id<MTLBuffer> outputsBuffer = (id<MTLBuffer>)TF_TensorData(outputs);

              id<MTLComputeCommandEncoder> encoder = commandBuffer.computeCommandEncoder;

              [encoder setComputePipelineState:pipeline];

              [encoder setBuffer:inputsBuffer offset:0 atIndex:0];
              [encoder setBuffer:embeddingsBuffer offset:0 atIndex:1];
              [encoder setBuffer:offsetsBuffer offset:0 atIndex:2];
              [encoder setBuffer:outputsBuffer offset:0 atIndex:3];

              [encoder setBytes:&B length:sizeof(B) atIndex:4];
              [encoder setBytes:&D length:sizeof(D) atIndex:5];
              [encoder setBytes:&C length:sizeof(C) atIndex:6];
              [encoder setBytes:&L length:sizeof(L) atIndex:7];

              // (ceil(B / 256), L, 1)  | (256, 1, 1)

              int threadsPerGroup = 256;
              int numInputPerGroup = ceil(float(B) / float(threadsPerGroup));


              MTLSize threadgroupsPerGrid = MTLSizeMake(numInputPerGroup, L, 1);
              MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
              [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

              [encoder endEncoding];
              [metalStream commit];
          }
        });
    }

    TF_DeleteTensor(inputs);
    TF_DeleteTensor(embeddings);
    TF_DeleteTensor(hashmap_offsets);
    TF_DeleteTensor(outputs);
    TF_DeleteStatus(status);
     */
}


void RegisterNearestNeighboursKernels(const char* device_type) {
    std::string opName("NearestNeighbours");
    auto* builder = TF_NewKernelBuilder("NearestNeighbours", device_type,
                                        &MetalNearestNeighboursOp_Create,
                                        &MetalNearestNeighboursOp_Compute,
                                        &MetalNearestNeighboursOp_Delete);

    TF_Status* status = TF_NewStatus();
    if (TF_OK != TF_GetCode(status))
        std::cout << " Error while registering " << opName << " kernel";
    TF_RegisterKernelBuilder((opName + "Op").c_str(), builder, status);
    if (TF_OK != TF_GetCode(status))
        std::cout << " Error while registering " << opName << " kernel";
    TF_DeleteStatus(status);
}



// Instantiate the kernels.
class InitPlugin {
   public:
    InitPlugin() {
        RegisterNearestNeighboursKernels("GPU");
    }
};

InitPlugin gInitPlugin;
