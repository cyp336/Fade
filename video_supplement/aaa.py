import cv2
import os

# 设置文件夹路径
folder_path = "/home/whl/workspace/cogvideo_edit/video_supplement"  # 修改为你的文件夹路径

# 获取文件夹下所有的.mp4文件
video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

# 循环处理每个视频文件
for video_file in video_files:
    # 读取视频文件
    video_path = os.path.join(folder_path, video_file)
    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建一个文件夹保存每帧
    output_folder = os.path.join(folder_path, os.path.splitext(video_file)[0])  # 用视频名作为文件夹名
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 逐帧处理视频
    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # 调整每帧的分辨率
        resized_frame = cv2.resize(frame, (512, 512))

        # 保存每帧为PNG文件
        output_frame_path = os.path.join(output_folder, f"frame_{frame_num:04d}.png")
        cv2.imwrite(output_frame_path, resized_frame)

    # 释放资源
    cap.release()

    print(f"视频 {video_file} 的每帧已保存至文件夹 {output_folder}")

print("所有视频处理完毕")
