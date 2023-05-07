#include <filesystem>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


@protocol TF_MetalStream
- (dispatch_queue_t)queue;

- (id <MTLCommandBuffer>)currentCommandBuffer;

- (void)commit;

- (void)commitAndWait;
@end


bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size()) {
    return false;
  }
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

std::optional <std::string> locate_metal_lib(std::string const &root) {
  using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
  auto installation_root = std::filesystem::path(root);
  for (const auto &dirEntry: recursive_directory_iterator(installation_root)) {
    if (ends_with(dirEntry.path().string(), "_nearest_neighbours.metallib")) {
      std::cout << "Found metallib at: " << dirEntry.path() << std::endl;
      return dirEntry.path().string();
    }
  }
  return nullptr;
}


// The singleton class for kernel library.
class KernelLibrarySingleton {
public:
  static KernelLibrarySingleton &getInstance() {
    if (sInstance == nullptr) {
      sInstance = new KernelLibrarySingleton();

      NSString *bundlePath = [[NSBundle mainBundle] resourcePath];
      NSString *parentPath = [bundlePath stringByDeletingLastPathComponent];
      auto parent_path_str = [parentPath cString];

      auto lib_path = locate_metal_lib(std::string(parent_path_str) + "/lib");
      if (lib_path->empty()) {
        lib_path = locate_metal_lib(std::filesystem::current_path().string());
      }
      if (lib_path->empty()) {
        std::cerr << "Failed to find metallib" << std::endl;
        std::abort();
      }

      @autoreleasepool {
        NSString *libraryFile = [NSString stringWithUTF8String:lib_path.value().c_str()];
        id <MTLDevice> device = MTLCreateSystemDefaultDevice();

        NSError *error = nil;
        NSURL *libraryUrl = [NSURL URLWithString:libraryFile];
        library = [device newLibraryWithURL:libraryUrl error:&error];

        if (!library) {
          printf("Compilation error: %s\n", [[error description] UTF8String]);
          std::abort();
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

std::vector<int> getShape(TF_Tensor *tensor) {
  std::vector<int> shape;
  const int dimensionCount = TF_NumDims(tensor);
  shape.resize(dimensionCount);
  for (int dim = 0; dim < dimensionCount; dim++) {
    shape[dim] = static_cast<int>(TF_Dim(tensor, dim));
  }
  return shape;
}


typedef struct NearestNeighboursOp {
} NearestNeighboursOp;

static void *NearestNeighboursOp_Create(TF_OpKernelConstruction *ctx) {
  return static_cast<void *>(new NearestNeighboursOp);
}

static void NearestNeighboursOp_Delete(void *kernel) {
  delete static_cast<NearestNeighboursOp *>(kernel);
}


typedef struct NearestNeighboursIndexesOp {
} NearestNeighboursIndexesOp;

static void *NearestNeighboursIndexesOp_Create(TF_OpKernelConstruction *ctx) {
  return static_cast<void *>(new NearestNeighboursIndexesOp);
}

static void NearestNeighboursIndexesOp_Delete(void *kernel) {
  delete static_cast<NearestNeighboursIndexesOp *>(kernel);
}

static void NearestNeighboursOp_Compute(void *kernel, TF_OpKernelContext *ctx) {

  TF_Status *status = TF_NewStatus();

  TF_Tensor *token_embeddings = nullptr;
  TF_GetInput(ctx, 0, &token_embeddings, status);

  TF_Tensor *embedding_matrix = nullptr;
  TF_GetInput(ctx, 1, &embedding_matrix, status);

  TF_DataType dataType = TF_TensorType(token_embeddings);

  const std::vector<int> embeddings_shape = getShape(token_embeddings);
  const auto ndim = embeddings_shape.size();
  switch (ndim) {
    case 1:
      break;
    case 2:
      break;
    case 3:
      break;
    default:
      break;
  }


  TF_DeleteTensor(token_embeddings);
  TF_DeleteTensor(embedding_matrix);
  TF_DeleteTensor(outputs);
  TF_DeleteStatus(status);
}


static void NearestNeighboursIndexesOp_Compute(void *kernel, TF_OpKernelContext *ctx) {

  TF_Status *status = TF_NewStatus();

  TF_Tensor *token_embeddings = nullptr;
  TF_GetInput(ctx, 0, &token_embeddings, status);

  TF_Tensor *embedding_matrix = nullptr;
  TF_GetInput(ctx, 1, &embedding_matrix, status);

  TF_DataType dataType = TF_TensorType(token_embeddings);

  const std::vector<int> embeddings_shape = getShape(token_embeddings);

  const auto ndim = embeddings_shape.size();
  const auto vocab_size = getShape(embedding_matrix)[0];
  const auto embedding_dim = getShape(embedding_matrix)[1];

  switch (ndim) {
    case 1: {
      TF_Tensor *outputs = TF_AllocateOutput(
        ctx, 0, dataType, {1,}, 1, 1, status
      );
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

            function = [[library newFunctionWithName:@"nearest_neighbours_indexes_1d"] autorelease];

            id <MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            assert(pipeline);

            auto inputsBuffer = (id <MTLBuffer>) TF_TensorData(token_embeddings);
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

            MTLSize threadgroupsPerGrid = MTLSizeMake(1, 1, 1);
            MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

            [encoder endEncoding];
            [metalStream commitAndWait];
          }
        });
      }
      TF_DeleteTensor(outputs);
      break;
    }
    case 2: {
      // TODO verify sizes
      TF_Tensor *outputs = TF_AllocateOutput(
        ctx, 0, dataType, (int64_t *) embeddings_shape.data(), 0, 1, status
      );
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

            function = [[library newFunctionWithName:@"nearest_neighbours_indexes_2d"] autorelease];

            id <MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            assert(pipeline);

            auto inputsBuffer = (id <MTLBuffer>) TF_TensorData(token_embeddings);
            id <MTLBuffer> embeddingsBuffer = (id <MTLBuffer>) TF_TensorData(embedding_matrix);
            id <MTLBuffer> outputsBuffer = (id <MTLBuffer>) TF_TensorData(outputs);

            id <MTLComputeCommandEncoder> encoder = commandBuffer.computeCommandEncoder;

            [encoder setComputePipelineState:pipeline];

            [encoder setBuffer:inputsBuffer offset:0 atIndex:0];
            [encoder setBuffer:embeddingsBuffer offset:0 atIndex:1];
            [encoder setBuffer:outputsBuffer offset:0 atIndex:2];
            [encoder setBytes:&vocab_size length:sizeof(int) atIndex:3];
            [encoder setBytes:&embedding_dim length:sizeof(int) atIndex:4];

            MTLSize threadgroupsPerGrid = MTLSizeMake(1, 1, 1);
            MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

            [encoder endEncoding];
            [metalStream commitAndWait];
          }
        });
      }
      TF_DeleteTensor(outputs);
      break;
    }
    case 3:
      break;
    default:
      break;
  }


  TF_DeleteTensor(token_embeddings);
  TF_DeleteTensor(embedding_matrix);
  TF_DeleteStatus(status);
}

template<typename T>
void RegisterKernel(const char *device_type) {
  std::string opName1("NearestNeighbours");
  std::string opName2("NearestNeighboursIndexes");

  auto *builder1 = TF_NewKernelBuilder(
    opName1.c_str(),
    device_type,
    &NearestNeighboursOp_Create,
    &NearestNeighboursOp_Compute,
    &NearestNeighboursOp_Delete
  );

  auto *builder2 = TF_NewKernelBuilder(
    opName2.c_str(),
    device_type,
    &NearestNeighboursIndexesOp_Create,
    &NearestNeighboursIndexesOp_Compute,
    &NearestNeighboursIndexesOp_Delete
  );

  TF_Status *status1 = TF_NewStatus();
  if (TF_OK != TF_GetCode(status1)) {
    std::cerr << " Error while registering " << opName1 << " kernel" << std::endl;
  }
  TF_RegisterKernelBuilder((opName1 + "Op").c_str(), builder1, status1);
  if (TF_OK != TF_GetCode(status1)) {
    std::cerr << " Error while registering " << opName1 << " kernel" << std::endl;
  }

  TF_Status *status2 = TF_NewStatus();
  if (TF_OK != TF_GetCode(status2)) {
    std::cerr << " Error while registering " << opName2 << " kernel" << std::endl;
  }
  TF_RegisterKernelBuilder((opName2 + "Op").c_str(), builder2, status2);
  if (TF_OK != TF_GetCode(status2)) {
    std::cerr << " Error while registering " << opName2 << " kernel" << std::endl;
  }


  TF_DeleteStatus(status1);
  TF_DeleteStatus(status2);
}

class InitPlugin {
public:
  InitPlugin() {
    id <MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
      RegisterKernel<float>("GPU");
    }
  }
};

InitPlugin gInitPlugin;