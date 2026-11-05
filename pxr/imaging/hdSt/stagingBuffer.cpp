//
// Copyright 2021 Pixar
//
// Licensed under the terms set forth in the LICENSE.txt file available at
// https://openusd.org/license.
//
#include "pxr/imaging/hdSt/resourceRegistry.h"
#include "pxr/imaging/hdSt/stagingBuffer.h"

#include "pxr/imaging/hgi/blitCmds.h"
#include "pxr/imaging/hgi/capabilities.h"
#include "pxr/imaging/hgi/hgi.h"

#include "pxr/base/tf/envSetting.h"

#include <algorithm>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_ENV_SETTING(HDST_MAX_STAGING_BUFFER_SIZE, 64 * 1024 * 1024,
    "Maximum bytes in a staging batch or direct buffer upload (1..1 GiB)");

namespace {
size_t
_GetMaxStagingBufferSize()
{
    static const size_t limit = []() -> size_t {
        const int configured = TfGetEnvSetting(HDST_MAX_STAGING_BUFFER_SIZE);
        if (configured <= 0 || configured > 1024 * 1024 * 1024) {
            TF_WARN("Invalid HDST_MAX_STAGING_BUFFER_SIZE; using 64 MiB");
            return 64 * 1024 * 1024;
        }
        return static_cast<size_t>(configured);
    }();
    return limit;
}
} // anonymous namespace


HdStStagingBuffer::HdStStagingBuffer(HdStResourceRegistry *resourceRegistry)
    : _resourceRegistry(resourceRegistry)
    , _head(0)
    , _capacity(0)
    , _activeSlot(0)
{
    _isUma = resourceRegistry->GetHgi()->GetCapabilities()->
                          IsSet(HgiDeviceCapabilitiesBitsUnifiedMemory);
}

HdStStagingBuffer::~HdStStagingBuffer()
{
    Deallocate();
}

void
HdStStagingBuffer::Deallocate()
{
    Hgi* hgi = _resourceRegistry->GetHgi();

    for (size_t i = 0; i < MULTIBUFFERING; ++i) {
        if (_handles[i]) {
            hgi->DestroyBuffer(&_handles[i]);
        }
    }

    _capacity = 0;
    _activeSlot = 0;
}

void
HdStStagingBuffer::Resize(size_t capacity)
{
    // Only change the capacity if there aren't any queued copy operations.
    if (_head != 0) {
        TF_CODING_ERROR("Cannot change size of staging buffer during Commit");
        return;
    }

    _capacity = std::min(capacity, _GetMaxStagingBufferSize());
}

void
HdStStagingBuffer::StageCopy(HgiBufferCpuToGpuOp const &copyOp)
{
    if (copyOp.byteSize == 0 ||
        !copyOp.cpuSourceBuffer ||
        !copyOp.gpuDestinationBuffer)
    {
        return;
    }

    // When the to-be-copied data is 'large' doing the extra memcpy into the
    // stating buffer to avoid many small GPU buffer upload can be more
    // expensive than just submitting the CPU to GPU copy operation directly.
    // The value of 'queueThreshold' is estimated (when is the extra memcpy
    // into the staging buffer slower than immediately issuing a gpu upload)
    static constexpr int queueThreshold = 512 * 1024;
    const size_t maxBatchSize = _GetMaxStagingBufferSize();
    if (_isUma && copyOp.byteSize > maxBatchSize) {
        // UMA must keep CPU writes in owned staging memory until the GPU
        // consumes them. Split oversized sources without writing directly
        // into a final buffer that an earlier frame may still be reading.
        for (size_t offset = 0; offset < copyOp.byteSize;) {
            HgiBufferCpuToGpuOp chunk = copyOp;
            chunk.sourceByteOffset += offset;
            chunk.destinationByteOffset += offset;
            chunk.byteSize = std::min(maxBatchSize, copyOp.byteSize - offset);
            StageCopy(chunk);
            offset += chunk.byteSize;
        }
        return;
    }
    if (!_isUma &&
        (copyOp.byteSize > queueThreshold ||
         copyOp.byteSize > maxBatchSize)) {
        // Preserve the direct-upload fast path, but never issue a giant
        // individual transfer. Sources stay alive until registry submission.
        HgiBlitCmds* blitCmds = _resourceRegistry->GetGlobalBlitCmds();
        for (size_t offset = 0; offset < copyOp.byteSize;) {
            HgiBufferCpuToGpuOp chunk = copyOp;
            chunk.sourceByteOffset += offset;
            chunk.destinationByteOffset += offset;
            chunk.byteSize = std::min(maxBatchSize, copyOp.byteSize - offset);
            blitCmds->CopyBufferCpuToGpu(chunk);
            offset += chunk.byteSize;
        }
        return;
    }

    // Resize receives an estimate. Resolved sources may require more space
    // than estimated, so grow within the batch limit instead of dropping data.
    const size_t requiredCapacity = std::max(_capacity, copyOp.byteSize);
    if (_head > requiredCapacity - copyOp.byteSize ||
        requiredCapacity != _capacity) {
        Flush();
        // Flush only records commands. Complete their execution before
        // reusing CPU staging bytes or a UMA buffer consumed by those copies.
        _resourceRegistry->SubmitBlitWork(HgiSubmitWaitTypeWaitUntilCompleted);
    }
    _capacity = requiredCapacity;

    HgiBufferHandle buffer = _handles[_activeSlot];
    constexpr size_t recoveryRatio = 4;

    // If there is no buffer in the active slot or it is either too small or
    // substantially larger than the required size, recreate it.
    if (!buffer ||
        buffer->GetDescriptor().byteSize < _capacity ||
        buffer->GetDescriptor().byteSize > _capacity * recoveryRatio) {
        HgiBufferDesc bufferDesc;
        bufferDesc.debugName = "HdStStagingBuffer";
        bufferDesc.byteSize = _capacity;
        bufferDesc.usage = HgiBufferUsageUpload;

        Hgi* hgi = _resourceRegistry->GetHgi();

        if (buffer) {
            hgi->DestroyBuffer(&buffer);
        }

        _handles[_activeSlot] = hgi->CreateBuffer(bufferDesc);
        buffer = _handles[_activeSlot];
    }

    size_t capacity = buffer->GetDescriptor().byteSize;
    uint8_t *cpuStaging = static_cast<uint8_t*>(buffer->GetCPUStagingAddress());

    if (TF_VERIFY(_head + copyOp.byteSize <= capacity)) {
        // Copy source into the staging buffer.
        char const *sourceData =
            static_cast<char const *>(copyOp.cpuSourceBuffer);
        memcpy(cpuStaging + _head,
               sourceData + copyOp.sourceByteOffset,
               copyOp.byteSize);

        bool aggregated = false;

        // If this copy is contiguous with last staged one then aggregate them.
        if (!_gpuCopyOps.empty()) {
            HgiBufferGpuToGpuOp &lastCopy = _gpuCopyOps.back();
            size_t lastCopyEnd = lastCopy.destinationByteOffset
                               + lastCopy.byteSize;

            if (lastCopy.gpuDestinationBuffer == copyOp.gpuDestinationBuffer &&
                lastCopyEnd == copyOp.destinationByteOffset) {
                lastCopy.byteSize += copyOp.byteSize;
                aggregated = true;
            }
        }

        if (!aggregated) {
            // Create a GPU to GPU blit operation to do the final copy.
            HgiBufferGpuToGpuOp gpuCopy;

            gpuCopy.gpuSourceBuffer = buffer;
            gpuCopy.sourceByteOffset = _head;
            gpuCopy.byteSize = copyOp.byteSize;
            gpuCopy.gpuDestinationBuffer = copyOp.gpuDestinationBuffer;
            gpuCopy.destinationByteOffset = copyOp.destinationByteOffset;

            _gpuCopyOps.push_back(gpuCopy);
        }

        _head += copyOp.byteSize;
    }
}

void
HdStStagingBuffer::Flush()
{
    if (_head == 0) {
        _gpuCopyOps.clear();
        return;
    }

    HgiBlitCmds* blitCmds = _resourceRegistry->GetGlobalBlitCmds();

    blitCmds->PushDebugGroup(__ARCH_PRETTY_FUNCTION__);

    if (!_isUma) {
        // If this isn't UMA then blit the staging buffer to GPU.
        HgiBufferCpuToGpuOp op;
        HgiBufferHandle buffer = _handles[_activeSlot];
        uint8_t* const cpuStaging = static_cast<uint8_t* const>(
            buffer->GetCPUStagingAddress());

        op.cpuSourceBuffer = cpuStaging;
        op.sourceByteOffset = 0;
        op.gpuDestinationBuffer = buffer;
        op.destinationByteOffset = 0;
        op.byteSize = _head;
        blitCmds->CopyBufferCpuToGpu(op);
    }

    blitCmds->InsertMemoryBarrier(HgiMemoryBarrierAll);
    for (auto const &copyOp : _gpuCopyOps) {
        blitCmds->CopyBufferGpuToGpu(copyOp);
    }

    blitCmds->PopDebugGroup();

    _gpuCopyOps.clear();
    _head = 0;

    if (_isUma) {
        _activeSlot++;
        _activeSlot = (_activeSlot < MULTIBUFFERING) ? _activeSlot : 0;
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
