#include "op.cuh"

class FillZeroOp : public PreloadOpBase {
   public:
    FillZeroOp() {
        //
    }

    __device__ void operator()() {
        //
    }
};

class PreloadOpBase : public OpBase {};