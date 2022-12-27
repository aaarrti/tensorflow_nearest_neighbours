#include <filesystem>
#include <dlfcn.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/c/kernels.h"


@protocol TF_MetalStream

- (dispatch_queue_t) queue;
- (id <MTLCommandBuffer>) currentCommandBuffer;
- (void) commit;
- (void) commitAndWait;

@end

// The singleton class for kernel library.
class KernelLibrarySingleton {
public:
  static KernelLibrarySingleton &getInstance() {
    if (sInstance == nullptr) {
      sInstance = new KernelLibrarySingleton();

      printf("Loading kernel library...\n");

      @autoreleasepool {
        NSString *libraryFile = @"_nearest_neighbours.metallib";
        id <MTLDevice> device = MTLCreateSystemDefaultDevice();

        NSError *error = nil;
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
  static id <MTLLibrary> library;

private:
  KernelLibrarySingleton() {}

  static KernelLibrarySingleton *sInstance;
};

KernelLibrarySingleton *KernelLibrarySingleton::sInstance = nullptr;
id <MTLLibrary> KernelLibrarySingleton::library = nil;

std::vector<int64_t> getShape(TF_Tensor *tensor) {
  std::vector<int64_t> shape;
  const int dimensionCount = TF_NumDims(tensor);
  shape.resize(dimensionCount);
  for (int dim = 0; dim < dimensionCount; dim++) {
    shape[dim] = TF_Dim(tensor, dim);
  }
  return shape;
}


typedef struct NearestNeighboursOp {} NearestNeighboursOp;

static void *NearestNeighboursOp_Create(TF_OpKernelConstruction *ctx) {
  return static_cast<void *>(new NearestNeighboursOp);
}

static void NearestNeighboursOp_Delete(void *kernel) {
  delete static_cast<NearestNeighboursOp *>(kernel);
}

static void NearestNeighboursOp_Compute(void *kernel, TF_OpKernelContext *ctx) {
  #ifdef DEBUG
  printf("----------Metal Kernel----------\n");
  #endif

  TF_Status *status = TF_NewStatus();

  TF_Tensor *token_embeddings = nullptr;
  TF_GetInput(ctx, 0, &token_embeddings, status);

  TF_Tensor *embedding_matrix = nullptr;
  TF_GetInput(ctx, 1, &embedding_matrix, status);

  TF_DataType dataType = TF_TensorType(token_embeddings);

  std::vector<int64_t> embeddings_shape = getShape(token_embeddings);
  const int vocab_size = getShape(embedding_matrix)[0];
  const int batch_size = embeddings_shape[0];
  const int num_tokens = embeddings_shape[1];
  const int embedding_dim = embeddings_shape[2];

  TF_Tensor *outputs = TF_AllocateOutput(ctx, 0, dataType, (int64_t *) embeddings_shape.data(), embeddings_shape.size(),
                                         0, status);

  if (TF_GetCode(status) != TF_OK) {
    printf("allocation failed: %s\n", TF_Message(status));
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteTensor(token_embeddings);
    TF_DeleteTensor(embedding_matrix);
    TF_DeleteTensor(outputs);
    TF_DeleteStatus(status);
    return;
  }

  @autoreleasepool {

    id <TF_MetalStream> metalStream = (id <TF_MetalStream>) (TF_GetStream(ctx, status));

    if (TF_GetCode(status) != TF_OK) {
      printf("no stream was found: %s\n", TF_Message(status));
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteTensor(token_embeddings);
      TF_DeleteTensor(embedding_matrix);
      TF_DeleteTensor(outputs);
      TF_DeleteStatus(status);
      return;
    }

    dispatch_sync(metalStream.queue, ^() {
      @autoreleasepool {
        id <MTLCommandBuffer> commandBuffer = metalStream.currentCommandBuffer;
        id <MTLDevice> device = commandBuffer.device;

        NSError *error = nil;
        id <MTLLibrary> library = KernelLibrarySingleton::getInstance().library;

        id <MTLFunction> function = nil;

        function = [[library newFunctionWithName:@"nearest_neighbours"] autorelease];

        id <MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        assert(pipeline);

        id <MTLBuffer> inputsBuffer = (id <MTLBuffer>) TF_TensorData(token_embeddings);
        id <MTLBuffer> embeddingsBuffer = (id <MTLBuffer>) TF_TensorData(embedding_matrix);
        id <MTLBuffer> outputsBuffer = (id <MTLBuffer>) TF_TensorData(outputs);

        id <MTLComputeCommandEncoder> encoder = commandBuffer.computeCommandEncoder;

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:inputsBuffer offset:0 atIndex:0];
        [encoder setBuffer:embeddingsBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputsBuffer offset:0 atIndex:2];

        [encoder setBytes:&num_tokens length:sizeof(int) atIndex:3];
        [encoder setBytes:&vocab_size length:sizeof(int) atIndex:4];
        [encoder setBytes:&embedding_dim length:sizeof(int) atIndex:5];

        MTLSize threadgroupsPerGrid = MTLSizeMake(batch_size, num_tokens, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        [encoder endEncoding];
        [metalStream commitAndWait];
      }
    });
  }

  TF_DeleteTensor(token_embeddings);
  TF_DeleteTensor(embedding_matrix);
  TF_DeleteTensor(outputs);
  TF_DeleteStatus(status);
}

template<typename T>
void RegisterKernel(const char *device_type) {
  std::string opName("NearestNeighbours");

  auto *builder = TF_NewKernelBuilder("NearestNeighbours", device_type, &NearestNeighboursOp_Create,
                                      &NearestNeighboursOp_Compute, &NearestNeighboursOp_Delete);

  TF_Status *status = TF_NewStatus();
  if (TF_OK != TF_GetCode(status))
    std::cout << " Error while registering " << opName << " kernel";
  TF_RegisterKernelBuilder((opName + "Op").c_str(), builder, status);
  if (TF_OK != TF_GetCode(status))
    std::cout << " Error while registering " << opName << " kernel";
  TF_DeleteStatus(status);
}


class InitPlugin {
public:
  InitPlugin() {
    RegisterKernel<float>("GPU");
  }
};

InitPlugin gInitPlugin;