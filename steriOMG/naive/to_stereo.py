import os
import sys
import numpy as np
import ffmpeg
from numpy.typing import NDArray
from tqdm import tqdm

from funcs import get_video_info

def create_side_by_side_video(video_buffer: NDArray, max_shift: int):
    num_frames, height, width, _ = video_buffer.shape
    single_video_width = width // 2

    pg = tqdm(total=num_frames*height, unit="pixel")
    pg.set_description('processing chunk')
    
    depth_map = video_buffer[:, :, single_video_width:, 0].astype(np.float32)
    min_depth, max_depth = np.min(depth_map), np.max(depth_map)
    depth_range = max_depth - min_depth
    
    # Precalculate the disparity map using the depth map buffer
    depth_map = (depth_map - min_depth) / depth_range * max_shift.astype(np.int32)
    
    for i in range(num_frames):
        for y in range(height):
            shifts = depth_map[i, y, :]
            shifted_xs = np.clip(np.arange(single_video_width) + shifts, 0, single_video_width - 1)
            for x in range(single_video_width):
                video_buffer[i, y, x + single_video_width, :] = video_buffer[i, y, shifted_xs[x], :]

            # video_buffer[i, y, single_video_width:, :] = video_buffer[i, y, shifted_xs, :]
            
            pg.update(1)

    pg.close()
    
    return video_buffer


def process_video(input_path, output_path, max_shift):
    video_info = get_video_info(input_path)
    output_width = video_info.width

    CHUNK_SIZE = 5 * 1024 * 1024 * 1024  # GB
    CHUNK_FRAMES = CHUNK_SIZE // (output_width * video_info.height * 3)

    vbuffer = np.zeros((CHUNK_FRAMES, video_info.height, output_width, 3), dtype=np.uint8)

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
    )
    in_original = ffmpeg.input(
        input_path,
        thread_queue_size=8192,
        vn=None,
        sn=None,
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

    # pg = tqdm(total=3, unit="chunk")
    # pg.set_description('converting to stereo')

    # for chunk_start in range(0, video_info.num_frames*2, CHUNK_FRAMES):
    #     chunk_end = min(chunk_start + CHUNK_FRAMES, video_info.num_frames)
    #     chunk_frames = chunk_end - chunk_start

    #     pg.set_description('reading buffer')
    #     for i in range(chunk_frames):
    #         in_bytes = process_input.stdout.read(video_info.width * video_info.height * 3)
    #         if not in_bytes:
    #             break

    #         vbuffer[i, :, :video_info.width, :] = np.frombuffer(in_bytes, np.uint8).reshape(
    #             [video_info.height, video_info.width, 3]
    #         )

    #     pg.set_description('processsing buffer')
    #     vbuffer[:chunk_frames] = create_side_by_side_video(vbuffer[:chunk_frames], max_shift)

    #     pg.set_description('writing buffer')
    #     for frame in vbuffer[:chunk_frames]:
    #         process_output.stdin.write(frame.tobytes())

    #     pg.update(1)
    # pg.close()


    progress_bar = tqdm(total=video_info.num_frames, unit="frames")

    for chunk_start in range(0, video_info.num_frames, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, video_info.num_frames)
        chunk_frames = chunk_end - chunk_start

        progress_bar.set_description('read')
        for i in range(chunk_frames):
            in_bytes = process_input.stdout.read(video_info.width * video_info.height * 3)
            if not in_bytes:
                break

            vbuffer[i, :, :video_info.width, :] = np.frombuffer(in_bytes, np.uint8).reshape(
                [video_info.height, video_info.width, 3]
            )

        progress_bar.set_description('process')
        vbuffer[:chunk_frames] = create_side_by_side_video(vbuffer[:chunk_frames], max_shift)

        progress_bar.set_description('write')
        for frame in vbuffer[:chunk_frames]:
            # try:
            #     process_output.stdin.write(frame.tobytes())
            # except BrokenPipeError:
            #     print("Broken pipe error occurred. Subprocess may have terminated unexpectedly.")
            # finally:
            #     process_output.stdin.close()
            #     process_output.wait()
            process_output.stdin.write(frame.tobytes())
            progress_bar.update(1)

    process_output.stdin.close()
    process_output.wait()
    progress_bar.close()

    process_output.stdin.close()
    process_output.wait()


def main():
    os.makedirs('./build/sbs', exist_ok=True)

    max_shift = 21

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "test.mp4"

    filepath = f"./build/depth/{filename}"
    output_path = f"./build/sbs/88qv-{max_shift}-{filename}"

    process_video(filepath, output_path, max_shift)


if __name__ == "__main__": main()
