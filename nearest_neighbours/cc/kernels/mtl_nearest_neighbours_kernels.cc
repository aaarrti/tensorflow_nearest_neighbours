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


std::vector<int> getShape(TF_Tensor* tensor) {
    std::vector<int> shape;
    const int dimensionCount = TF_NumDims(tensor);
    shape.resize(dimensionCount);
    for (int dim = 0; dim < dimensionCount; dim++) {
        shape[dim] = TF_Dim(tensor, dim);
    }
    return shape;
}


static void MetalNearestNeighboursOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
    auto* k = static_cast<MetalNearestNeighboursOp *>(kernel);

    TF_Status* status = TF_NewStatus();

    TF_Tensor* embeddings_batch = nullptr;
    TF_GetInput(ctx, 0, &embeddings_batch, status);

    TF_Tensor* embedding_matrix = nullptr;
    TF_GetInput(ctx, 1, &embedding_matrix, status);

    TF_DataType dataType = TF_TensorType(embeddings_batch);


    const auto shape = getShape(embeddings_batch);
    const auto batch_size = shape[0];
    const auto sequence_length = shape[1];
    const auto embedding_dim = shape[2];

    const auto vocab_size = getShape(embedding_matrix)[0];


    std::vector<int> output_shape{batch_size, sequence_length, embedding_dim};

    TF_Tensor* outputs = TF_AllocateOutput(ctx, 0, dataType, (int64_t*)output_shape.data(), output_shape.size(), 0, status);

    if (TF_GetCode(status) != TF_OK) {
        printf("allocation failed: %s\n", TF_Message(status));
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteTensor(embeddings_batch);
        TF_DeleteTensor(embedding_matrix);
        TF_DeleteTensor(outputs);
        TF_DeleteStatus(status);
        return;
    }


    @autoreleasepool {

        id<TF_MetalStream> metalStream = (id<TF_MetalStream>)(TF_GetStream(ctx, status));

        if (TF_GetCode(status) != TF_OK) {
            printf("no stream was found: %s\n", TF_Message(status));
            TF_OpKernelContext_Failure(ctx, status);
            TF_DeleteTensor(embeddings_batch);
            TF_DeleteTensor(embedding_matrix);
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


              id<MTLBuffer> arg1Buffer = (id<MTLBuffer>)TF_TensorData(embeddings_batch);
              id<MTLBuffer> arg2Buffer = (id<MTLBuffer>)TF_TensorData(embedding_matrix);
              id<MTLBuffer> outputsBuffer = (id<MTLBuffer>)TF_TensorData(outputs);

              id<MTLComputeCommandEncoder> encoder = commandBuffer.computeCommandEncoder;

              [encoder setComputePipelineState:pipeline];

              [encoder setBuffer:arg1Buffer offset:0 atIndex:0];
              [encoder setBuffer:arg2Buffer offset:0 atIndex:1];
              [encoder setBuffer:outputsBuffer offset:0 atIndex:3];

              [encoder setBytes:&batch_size length:sizeof(batch_size) atIndex:4];
              [encoder setBytes:&sequence_length length:sizeof(sequence_length) atIndex:5];
              [encoder setBytes:&embedding_dim length:sizeof(embedding_dim) atIndex:6];

              MTLSize threadgroupsPerGrid = MTLSizeMake(batch_size, sequence_length , 1);
              MTLSize threadsPerThreadgroup = MTLSizeMake(sequence_length, 1, 1);
              [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

              [encoder endEncoding];
              [metalStream commit];

          }
        });
    }

    TF_DeleteTensor(embeddings_batch);
    TF_DeleteTensor(embedding_matrix);
    TF_DeleteTensor(outputs);
    TF_DeleteStatus(status);

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
