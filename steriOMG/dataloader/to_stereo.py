import os
import sys
import ffmpeg
import cupy as cp
import numpy as np
import pycuda.driver as cuda

from tqdm import tqdm
from pycuda import compiler
from funcs import get_video_info

# def create_side_by_side_video(video_buffer: np.ndarray, max_shift: int):
#     num_frames, height, width, _ = video_buffer.shape
#     single_video_width = width // 2
    
#     depth_map = video_buffer[:, :, single_video_width:, 0].astype(np.float32)
#     min_depth = np.min(depth_map)
#     max_depth = np.max(depth_map)
#     depth_range = max_depth - min_depth
    
#     # Precalculate the disparity map using the depth map buffer
#     depth_map -= min_depth
#     depth_map /= depth_range
#     depth_map *= max_shift
#     depth_map = depth_map.astype(np.int32)
    
#     for i in range(num_frames):
#         for y in range(height):
#             shifts = depth_map[i, y, :]
#             shifted_xs = np.clip(np.arange(single_video_width) + shifts, 0, single_video_width - 1)
#             video_buffer[i, y, single_video_width:, :] = video_buffer[i, y, shifted_xs, :]
    
#     return video_buffer


def create_side_by_side_video(video_buffer: cp.ndarray, max_shift: int):
    num_frames, height, width, _ = video_buffer.shape
    single_video_width = width // 2
    
    depth_map = video_buffer[:, :, single_video_width:, 0].astype(cp.float32)
    min_depth, max_depth = cp.min(depth_map), cp.max(depth_map)
    depth_range = max_depth - min_depth

    # Precalculate the disparity map
    depth_map = ((depth_map - min_depth) / depth_range * max_shift).astype(cp.int32)

    # Allocate GPU memory (already on GPU since video_buffer is CuPy)
    shifted_video_gpu = cp.empty_like(video_buffer)

    # Compile the kernel
    with open('shift_video.cu', 'r') as f:kernel_code = f.read()
    mod = compiler.SourceModule(kernel_code)
    shift_video = mod.get_function("shift_video")

    # Launch the kernel
    grid_size = (num_frames, height)
    block_size = (single_video_width, 1, 1)
    # shift_video(
    #     np.int32(num_frames), np.int32(height), np.int32(single_video_width), 
    #     np.int32(width), np.int32(max_shift), depth_map.data.ptr, 
    #     video_buffer.data.ptr, shifted_video_gpu.data.ptr, 
    #     block=block_size, grid=grid_size
    # )
    shift_video(
        video_buffer.data.ptr, np.int32(width), np.int32(height), np.int32(max_shift),
        block=block_size, grid=grid_size
    )

    return shifted_video_gpu


def process_video(input_path, output_path, max_shift):
    video_info = get_video_info(input_path)
    output_width = video_info.width

    CHUNK_SIZE = 20 * (1024**3) # GB
    CHUNK_FRAMES = CHUNK_SIZE // (output_width * video_info.height * 3)

    vbuffer = cp.empty((CHUNK_FRAMES, video_info.height, video_info.width, 3), dtype=cp.uint8)

    # Create buffers for input/output streams
    process_input = (
        ffmpeg.input(
            input_path,
            threads=0,
            thread_queue_size=8192,
        )
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet")
        .run_async(pipe_stdout=True)
    )

    in_modified = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s=f"{output_width}x{video_info.height}",
        framerate=video_info.framerate,
        thread_queue_size=8192,
        hwaccel="cuda",
    )
    
    in_original = ffmpeg.input(
        input_path,
        thread_queue_size=8192,
        vn=None,
        sn=None,
        hwaccel="cuda",
    )
    
    process_output = ffmpeg.output(
        in_modified,
        in_original,
        output_path,
        acodec="copy",
        threads=0,
        framerate=video_info.framerate,
        s=f"{output_width}x{video_info.height}",
        loglevel="quiet",
        **{'q:v': '88'}
    ).overwrite_output().run_async(pipe_stdin=True)

    progress_bar = tqdm(total=video_info.num_frames, unit="frames")

    # Iterate over batches of frames
    for chunk_start in range(0, video_info.num_frames, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, video_info.num_frames)
        chunk_frames = chunk_end - chunk_start

        # Use ffmpeg to read a chunk from the buffer into numpy array
        progress_bar.set_description('read')
        for i in range(chunk_frames):
            in_bytes = process_input.stdout.read(video_info.width * video_info.height * 3)
            if not in_bytes: break
            data_gpu = cp.frombuffer(in_bytes, dtype=cp.uint8).reshape(
                (video_info.height, video_info.width, 3)
            )
            vbuffer[i, :, :, :] = data_gpu

        # Compute the mapping
        progress_bar.set_description('process')
        vbuffer[:chunk_frames] = create_side_by_side_video(vbuffer[:chunk_frames], max_shift)

        # Write back to the buffer
        progress_bar.set_description('write')
        for frame in vbuffer[:chunk_frames]:
            process_output.stdin.write(frame.tobytes())
            progress_bar.update(1)

    process_output.stdin.close()
    process_output.wait()
    progress_bar.close()


def main():
    os.makedirs('./build/sbs', exist_ok=True)

    max_shift = 21
    DEVICE_ID = 0

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "test.mp4"

    filepath = f"./build/depth/{filename}"
    output_path = f"./build/sbs/88qv-{max_shift}-{filename}"

    try:
        cp.cuda.Device(DEVICE_ID).use()
        device = cuda.Device(DEVICE_ID)
        device.make_context()
        process_video(filepath, output_path, max_shift)
    finally:
        cuda.Context.pop()


if __name__ == "__main__": main()
