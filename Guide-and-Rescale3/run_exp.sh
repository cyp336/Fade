#!/bin/bash

# 定义 YAML 文件路径和 Python 文件路径
yaml_file="/home/cby/cogvideo_code/Guide-and-Rescale3/configs/ours_nonstyle_best_cogvideo.yaml"
python_file="/home/cby/cogvideo_code/Guide-and-Rescale3/guide_and_rescale_cogvideo.py"  # 替换成实际的 Python 文件路径

# 设置 self_attn_gs 和 app_gs 的值范围
self_attn_gs_values=($(seq 12000 -1000 1000))  # 设定自定义的 self_attn_gs 值
app_gs_values=($(seq 1 0.5 10))           # 设定自定义的 app_gs 值

# 双重循环：遍历 self_attn_gs 和 app_gs 的组合
for self_attn_gs in "${self_attn_gs_values[@]}"; do
    for app_gs in "${app_gs_values[@]}"; do
        # 使用 sed 修改 YAML 文件中的 self_attn_gs 和 app_gs 值
        sed -i "s/self_attn_gs: .*/self_attn_gs: ${self_attn_gs}/" "$yaml_file"
        sed -i "s/app_gs: .*/app_gs: ${app_gs}/" "$yaml_file"
        
        # 输出当前设置（可选）
        echo "Updated self_attn_gs to ${self_attn_gs} and app_gs to ${app_gs} in $yaml_file"
        
        # 运行 Python 文件
        python "$python_file"
    done
done
