import os
from os.path import exists, join
import argparse
import subprocess
from multiprocessing import Pool
import shutil
from pathlib import Path
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
from functools import partial
from tqdm import tqdm

def is_video_file(file_path):
    """检查文件是否为视频文件"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def compress_video(input_output_pair, size=224, fps=6):
    """压缩视频（调整FPS和尺寸）, follow E.T. Bench"""
    input_path, output_path = input_output_pair
    try:
        command = ['ffmpeg',
                  '-y',  
                  '-i', str(input_path),
                  '-filter:v',  
                  f"scale='if(gt(a,1),trunc(oh*a/2)*2,{size})':'if(gt(a,1),{size},trunc(ow*a/2)*2)'",
                  '-map', '0:v',  
                  '-r', str(fps),  
                  str(output_path)
                  ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:  
            print(f"压缩失败 {input_path}: {result.stderr.decode()}")  
        return result.returncode == 0
    except Exception as e:
        print(f"压缩失败 {input_path}: {e}")
        return False

def process_videos(input_root, output_dir, size=224, fps=6, num_workers=24):
    """处理所有视频文件"""
    if not os.path.exists(input_root):
        print(f"输入目录不存在: {input_root}")
        return 0, 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = []
    all_files = os.listdir(input_root)
    all_files = [os.path.join(input_root, x) for x in all_files]
    print(f"找到 {len(all_files)} 个文件，开始检查文件类型...")
    
    video_files = []
    for file_path in tqdm(all_files, desc="检查文件类型"):
        if is_video_file(file_path):
            video_files.append(file_path)
    
    print(f"其中有 {len(video_files)} 个视频文件需要处理")
    
    input_pairs = []
    for video_path in video_files:
        output_path = os.path.join(output_dir, f'{Path(video_path).stem}.mp4')
        input_pairs.append((video_path, output_path))
    
    print(f"需要处理的文件数: {len(input_pairs)}")
    
    compress = partial(compress_video, size=size, fps=fps)
    
    success_count = 0
    fail_count = 0
    
    with Pool(num_workers) as pool, tqdm(total=len(input_pairs), desc="处理视频") as pbar:
        for idx, success in enumerate(pool.imap_unordered(compress, input_pairs, chunksize=32)):
            if success:
                success_count += 1
            else:
                fail_count += 1
            pbar.update(1)
    
    print("压缩完成，处理失败的文件...")
    copy_count = 0
    for input_path, output_path in input_pairs:
        if exists(input_path):
            if not exists(output_path) or os.path.getsize(output_path) < 1:
                copy_count += 1
                shutil.copyfile(input_path, output_path)
                print(f"复制并替换文件: {output_path}")
    
    return success_count, fail_count, copy_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理视频文件：转换格式并压缩")
    parser.add_argument("--input_root", type=str, 
                       default="/storage/wenzheng/dataset/TimeChat/TimeIT/data/temporal_video_grounding/charades/Charades_v1", 
                       help="输入根目录")
    parser.add_argument("--output_dir", type=str, 
                       default="/storage/wenzheng/dataset/D2VLM_other_benchmark/charades/compressed_videos", 
                       help="输出目录")
    parser.add_argument("--size", type=int, default=224, help="短边尺寸")
    parser.add_argument("--fps", type=int, default=3, help="输出视频帧率")
    parser.add_argument("--num_workers", type=int, default=24, help="处理进程数")
    
    args = parser.parse_args()
    
    print("开始处理视频文件...")
    success, fail, copied = process_videos(
        args.input_root,
        args.output_dir,
        args.size,
        args.fps,
        args.num_workers
    )
    
    print("\n处理完成！")
    print(f"成功处理: {success} 个文件")
    print(f"处理失败: {fail} 个文件")
    print(f"直接复制: {copied} 个文件")
    print(f"输出目录: {args.output_dir}")