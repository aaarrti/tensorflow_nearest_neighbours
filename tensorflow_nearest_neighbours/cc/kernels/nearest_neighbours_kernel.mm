#include <filesystem>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/tsl/c/tsl_status_helper.h"


@protocol TF_MetalStream
- (dispatch_queue_t)queue;

- (id <MTLCommandBuffer>)currentCommandBuffer;

- (void)commit;

- (void)commitAndWait;
@end

// --------------------------------------------------------------
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
// ---------------------------------------------------------------------


bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size()) {
    return false;
  }
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

std::optional<std::string> locate_metal_lib(std::string const &root) {
  using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
  auto installation_root = std::filesystem::path(root);
  for (const auto &dirEntry: recursive_directory_iterator(installation_root)) {
    if (ends_with(dirEntry.path().string(), "_nearest_neighbours.metallib")) {
      std::cerr << "--------------------------------" << std::endl;
      std::cout << "Found metallib at: " << dirEntry.path() << std::endl;
      std::cerr << "--------------------------------" << std::endl;
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
        std::cerr << "--------------------------------" << std::endl;
        std::cerr << "Failed to find metal lib" << std::endl;
        std::cerr << "--------------------------------" << std::endl;
        std::abort();
      }

      @autoreleasepool {
        NSString *libraryFile = [NSString stringWithUTF8String:lib_path.value().c_str()];
        id <MTLDevice> device = MTLCreateSystemDefaultDevice();

        NSError *error = nil;
        NSURL *libraryUrl = [NSURL URLWithString:libraryFile];
        library = [device newLibraryWithURL:libraryUrl error:&error];

        if (!library) {
          std::cerr << "--------------------------------" << std::endl;
          printf("Compilation error: %s\n", [[error description] UTF8String]);
          std::cerr << "--------------------------------" << std::endl;
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
  for (int dim = 0; dim < dimensionCount; dim++) {
    shape.push_back(static_cast<int>(TF_Dim(tensor, dim)));
  }
  return shape;
}

// -----------------------------------------------------------------------------

static void launch_kernel(
  std::string name,
  TF_OpKernelContext *ctx,
  TF_Status *status,
  TF_Tensor *token_embeddings,
  TF_Tensor *embedding_matrix,
  TF_Tensor *outputs,
  const int num_tokens,
  const int vocab_size,
  const int embedding_dim,
  const int threadGroupsPerGrid_X,
  const int threadGroupsPerGrid_Y,
  const int threadGroupsPerGrid_Z,
  const int threadsPerThreadGroup_X,
  const int threadsPerThreadGroup_Y,
  const int threadsPerThreadGroup_Z
) {
  @autoreleasepool {
    id <TF_MetalStream> metalStream = (id <TF_MetalStream>) (TF_GetStream(ctx, status));

    dispatch_sync(metalStream.queue, ^() {
      @autoreleasepool {
        id <MTLCommandBuffer> commandBuffer = metalStream.currentCommandBuffer;
        id <MTLDevice> device = commandBuffer.device;
        NSError *error = nil;
        id <MTLLibrary> library = KernelLibrarySingleton::getInstance().library;
        id <MTLFunction> function = nil;
        function = [[library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]] autorelease];
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
        MTLSize threadGroupsPerGrid = MTLSizeMake(
          threadGroupsPerGrid_X,
          threadGroupsPerGrid_Y,
          threadGroupsPerGrid_Z
        );
        MTLSize threadsPerThreadGroup = MTLSizeMake(
          threadsPerThreadGroup_X,
          threadGroupsPerGrid_Y,
          threadGroupsPerGrid_Z
        );
        [encoder dispatchThreadgroups:threadGroupsPerGrid threadsPerThreadgroup:threadsPerThreadGroup];
        [encoder endEncoding];
        [metalStream commitAndWait];
      }
    });
  }
}

static void default_launch_kernel(
  std::string name,
  TF_OpKernelContext *ctx,
  TF_Status *status,
  TF_Tensor *token_embeddings,
  TF_Tensor *embedding_matrix,
  TF_Tensor *outputs,
  const int num_tokens,
  const int vocab_size,
  const int embedding_dim,
  const int threadGroupsPerGrid_X,
  const int threadGroupsPerGrid_Y
) {
  launch_kernel(
    name,
    ctx,
    status,
    token_embeddings,
    embedding_matrix,
    outputs,
    num_tokens,
    vocab_size,
    embedding_dim,
    threadGroupsPerGrid_X,
    threadGroupsPerGrid_Y,
    1,
    1,
    1,
    1
  );
}

// ------------------------------------------------------------------------------


static void NearestNeighboursOp_Compute(void *kernel, TF_OpKernelContext *ctx) {

  TF_Status *status = TF_NewStatus();

  TF_Tensor *token_embeddings = nullptr;
  TF_GetInput(ctx, 0, &token_embeddings, status);

  TF_Tensor *embedding_matrix = nullptr;
  TF_GetInput(ctx, 1, &embedding_matrix, status);

  TF_DataType dataType = TF_TensorType(token_embeddings);

  const std::vector<int> input_shape = getShape(token_embeddings);
  const std::vector<int> embedding_matrix_shape = getShape(embedding_matrix);
  const int vocab_size = embedding_matrix_shape[0];
  const int embedding_dim = embedding_matrix_shape[1];

  const auto ndim = input_shape.size();
  TF_Tensor *outputs = TF_AllocateOutput(
    ctx, 0, dataType, (int64_t *) input_shape.data(),
    input_shape.size(), 0, status
  );
  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "allocation failed: " << TF_Message(status) << std::endl;
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteTensor(token_embeddings);
    TF_DeleteTensor(embedding_matrix);
    TF_DeleteTensor(outputs);
    TF_DeleteStatus(status);
  }

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "ndim = " << ndim << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  switch (ndim) {
    case 1: {
      std::string name = "nearest_neighbours_1d";
      default_launch_kernel(
        name,
        ctx,
        status,
        token_embeddings,
        embedding_matrix,
        outputs,
        0,
        vocab_size,
        embedding_dim,
        1,
        1
      );
      break;
    }
    case 2: {
      std::string name = "nearest_neighbours_2d";
      const int num_tokens = input_shape[0];
      default_launch_kernel(
        name,
        ctx,
        status,
        token_embeddings,
        embedding_matrix,
        outputs,
        num_tokens,
        vocab_size,
        embedding_dim,
        num_tokens,
        1
      );
      break;
    }
    case 3: {
      std::string name = "nearest_neighbours_3d";
      const int batch_size = input_shape[0];
      const int num_tokens = input_shape[1];
      default_launch_kernel(
        name,
        ctx,
        status,
        token_embeddings,
        embedding_matrix,
        outputs,
        num_tokens,
        vocab_size,
        embedding_dim,
        batch_size,
        num_tokens
      );
      break;
    }
    default: {
      const auto error = tensorflow::errors::InvalidArgument("ndim = " + std::to_string(ndim));
      tsl::Set_TSL_Status_from_Status(status, error);
      TF_OpKernelContext_Failure(ctx, status);
    }
  }


  TF_DeleteTensor(token_embeddings);
  TF_DeleteTensor(embedding_matrix);
  TF_DeleteStatus(status);
  TF_DeleteTensor(outputs);
}


static void NearestNeighboursIndexesOp_Compute(void *kernel, TF_OpKernelContext *ctx) {

  TF_Status *status = TF_NewStatus();
  TF_Tensor *token_embeddings = nullptr;
  TF_GetInput(ctx, 0, &token_embeddings, status);
  TF_Tensor *embedding_matrix = nullptr;
  TF_GetInput(ctx, 1, &embedding_matrix, status);
  TF_DataType dataType = TF_TensorType(token_embeddings);


  const std::vector<int> input_shape = getShape(token_embeddings);
  const std::vector<int> embedding_matrix_shape = getShape(embedding_matrix);

  const auto ndim = input_shape.size();
  const auto vocab_size = embedding_matrix_shape[0];
  const auto embedding_dim = embedding_matrix_shape[1];

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "ndim = " << ndim << std::endl;
  std::cout << "-----------------------------------" << std::endl;



  switch (ndim) {
    case 1: {
      TF_Tensor *outputs = TF_AllocateOutput(ctx, 0, TF_INT32, {}, 0, 0, status);
      if (TF_GetCode(status) != TF_OK) {
        std::cerr << "allocation failed: " << TF_Message(status) << std::endl;
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteTensor(token_embeddings);
        TF_DeleteTensor(embedding_matrix);
        TF_DeleteTensor(outputs);
        TF_DeleteStatus(status);
      }
      std::string name = "nearest_neighbours_indexes_1d";
      default_launch_kernel(
        name,
        ctx,
        status,
        token_embeddings,
        embedding_matrix,
        outputs,
        0,
        vocab_size,
        embedding_dim,
        1,
        1
      );
      break;
    }
    case 2: {
      std::string name = "nearest_neighbours_indexes_3d";
      const int batch_size = input_shape[0];
      const int num_tokens = input_shape[1];
      const std::vector<int64_t> dims = std::vector<int64_t>{num_tokens};
      TF_Tensor *outputs = TF_AllocateOutput(
        ctx, 0, TF_INT32, dims.data(), 1, 0, status
      );
      if (TF_GetCode(status) != TF_OK) {
        std::cerr << "allocation failed: " << TF_Message(status) << std::endl;
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteTensor(token_embeddings);
        TF_DeleteTensor(embedding_matrix);
        TF_DeleteTensor(outputs);
        TF_DeleteStatus(status);
      }
      default_launch_kernel(
        name,
        ctx,
        status,
        token_embeddings,
        embedding_matrix,
        outputs,
        num_tokens,
        vocab_size,
        embedding_dim,
        1,
        num_tokens
      );
      break;
    }
    case 3: {
      std::string name = "nearest_neighbours_indexes_3d";
      const int batch_size = input_shape[0];
      const int num_tokens = input_shape[1];
      const std::vector<int64_t> dims = std::vector<int64_t>{batch_size, num_tokens};
      TF_Tensor *outputs = TF_AllocateOutput(
        ctx, 0, TF_INT32, dims.data(), 2, 0, status
      );
      if (TF_GetCode(status) != TF_OK) {
        std::cerr << "allocation failed: " << TF_Message(status) << std::endl;
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteTensor(token_embeddings);
        TF_DeleteTensor(embedding_matrix);
        TF_DeleteTensor(outputs);
        TF_DeleteStatus(status);
      }
      default_launch_kernel(
        name,
        ctx,
        status,
        token_embeddings,
        embedding_matrix,
        outputs,
        num_tokens,
        vocab_size,
        embedding_dim,
        batch_size,
        num_tokens
      );
      break;
    }
    default: {
      const auto error = tensorflow::errors::InvalidArgument("ndim = " + std::to_string(ndim));
      const auto error_status = TF_NewStatus();
      tsl::Set_TSL_Status_from_Status(error_status, error);
      TF_OpKernelContext_Failure(ctx, status);
    }
  }


  TF_DeleteTensor(token_embeddings);
  TF_DeleteTensor(embedding_matrix);
  TF_DeleteStatus(status);
}

class InitPlugin {
public:
  InitPlugin() {
    id <MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
      std::string opName1("NearestNeighbours");
      std::string opName2("NearestNeighboursIndexes");

      auto *builder1 = TF_NewKernelBuilder(
        opName1.c_str(),
        "GPU",
        &NearestNeighboursOp_Create,
        &NearestNeighboursOp_Compute,
        &NearestNeighboursOp_Delete
      );

      auto *builder2 = TF_NewKernelBuilder(
        opName2.c_str(),
        "GPU",
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
  }
};

InitPlugin gInitPlugin;