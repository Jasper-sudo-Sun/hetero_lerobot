#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

def main():
    # 文件路径
    file_path = "/mnt/hpfs/baaiei/qianpusun/lerobot/data/HuaihaiLyu/agilex_pour_tea/data/chunk-000/episode_000000.parquet"
    
    # 读取数据
    print(f"正在读取文件: {file_path}")
    df = pd.read_parquet(file_path)
    
    # 显示基本信息
    print("\n=== 数据基本信息 ===")
    print(f"数据形状: {df.shape}")
    print("\n列名:")
    print(df.columns.tolist())
    print("\n前5行数据:")
    print(df.head())

if __name__ == "__main__":
    main() 