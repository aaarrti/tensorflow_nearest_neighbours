#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <filesystem>
#include <sys/_types/_int32_t.h>
#include <dlfcn.h>

#include "tensorflow/c/kernels.h"
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

@protocol TF_MetalStream

- (dispatch_queue_t) queue;
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
        NSString* libraryFile = @"_nearest_neighbours.metallib";
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

std::vector<int64_t> getShape(TF_Tensor* tensor) {
  std::vector<int64_t> shape;
  const int dimensionCount = TF_NumDims(tensor);
  shape.resize(dimensionCount);
  for (int dim = 0; dim < dimensionCount; dim++) {
    shape[dim] = TF_Dim(tensor, dim);
  }
  return shape;
}

// The hash encode forward part.

typedef struct NearestNeighboursOp {} NearestNeighboursOp;

static void* NearestNeighboursOp_Create(TF_OpKernelConstruction* ctx) {
  static NearestNeighboursOp op;
  return static_cast<void*>(&op);
}

static void NearestNeighboursOp_Delete(void* kernel) {}

static void NearestNeighboursOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  printf("-------Metal Kernel--------\n");

}

  template <typename T>
  void RegisterKernel(const char* device_type) {
    std::string opName("NearestNeighbours");

    auto* builder = TF_NewKernelBuilder("NearestNeighbours", device_type, &NearestNeighboursOp_Create, &NearestNeighboursOp_Compute, &NearestNeighboursOp_Delete);

    TF_Status* status = TF_NewStatus();
    if (TF_OK != TF_GetCode(status))
      std::cout << " Error while registering " << opName << " kernel";
    TF_RegisterKernelBuilder((opName + "Op").c_str(), builder, status);
    if (TF_OK != TF_GetCode(status))
      std::cout << " Error while registering " << opName << " kernel";
    TF_DeleteStatus(status);
  }




  class InitPlugin {
    public:
      InitPlugin(){
          RegisterKernel<float>("GPU");
      }
  };

 InitPlugin gInitPlugin;