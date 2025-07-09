from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import h5py


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples where Y only contains traffic data (no time features)
    """
    num_samples, num_nodes = df.shape
    
    # 🚗 Raw traffic data (T, N, 1)
    traffic_data = np.expand_dims(df.values, axis=-1)
    
    # 📊 Build X features (traffic + time + day)
    x_feature_list = [traffic_data]
    
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        x_feature_list.append(time_in_day)
        
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        x_feature_list.append(dow_tiled)

    # X data: traffic + temporal features
    x_data = np.concatenate(x_feature_list, axis=-1)
    
    # 🎯 Y data: ONLY traffic data (no temporal features)
    y_data = traffic_data  # Shape: (T, N, 1)
    
    print(f"📊 Feature dimensions:")
    print(f"  X data shape: {x_data.shape} (traffic + temporal)")
    print(f"  Y data shape: {y_data.shape} (traffic only)")
    
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    
    for t in range(min_t, max_t):
        x.append(x_data[t + x_offsets, ...])
        y.append(y_data[t + y_offsets, ...])  # Y chỉ có traffic data
        
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    
    print(f"🎯 Final shapes:")
    print(f"  X: {x.shape} (samples, timesteps, nodes, features)")
    print(f"  Y: {y.shape} (samples, timesteps, nodes, 1)")
    
    return x, y


def load_h5_data(filename, data_key):
    """Load data from H5 file"""
    print(f"🔍 Loading data from {filename} with key: {data_key}")
    
    with h5py.File(filename, 'r') as f:
        print(f"📋 Available keys in file: {list(f.keys())}")
        
        if data_key not in f.keys():
            raise ValueError(f"Key '{data_key}' not found in file. Available: {list(f.keys())}")
        
        data = np.array(f[data_key])
        print(f"✅ Data loaded: shape = {data.shape}")
    
    T, N = data.shape
    
    if 'NYC' in filename:
        start_time = '2016-04-01 00:00:00'
    else:
        start_time = '2024-07-01 00:00:00'
    
    time_index = pd.date_range(start=start_time, periods=T, freq='30min')
    df = pd.DataFrame(data, index=time_index, columns=[f'node_{i}' for i in range(N)])
    
    print(f"📊 DataFrame created: {df.shape}")
    print(f"⏰ Time range: {df.index.min()} to {df.index.max()}")
    
    return df


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    
    # Load data
    df = load_h5_data(args.traffic_df_filename, args.data_key)
    
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + args.y_start), 1))
    
    print(f"🔧 X offsets: {x_offsets}")
    print(f"🎯 Y offsets: {y_offsets}")
    
    # Generate sequences
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_day_in_week=args.add_day_in_week,
    )

    # 🔥 CHIA DATASET THEO YÊU CẦU MỚI
    num_samples = x.shape[0]
    num_test = 672   # 672 samples cuối cho test
    num_val = 672    # 672 samples kề cuối cho val
    num_train = num_samples - num_test - num_val  # Phần còn lại cho train
    
    print(f"📊 Dataset split (fixed sizes):")
    print(f"  Total samples: {num_samples}")
    print(f"  Train: {num_train} samples (samples 0 to {num_train-1})")
    print(f"  Val: {num_val} samples (samples {num_train} to {num_train + num_val - 1})")
    print(f"  Test: {num_test} samples (samples {num_train + num_val} to {num_samples-1})")
    
    # Kiểm tra có đủ data không
    if num_train <= 0:
        raise ValueError(f"Not enough data! Total samples: {num_samples}, need at least {num_test + num_val} samples")
    
    # Chia data theo thứ tự thời gian
    x_train = x[:num_train]
    y_train = y[:num_train]
    
    x_val = x[num_train:num_train + num_val]
    y_val = y[num_train:num_train + num_val]
    
    x_test = x[num_train + num_val:]
    y_test = y[num_train + num_val:]

    # Verify split
    print(f"\n✅ Verification:")
    print(f"  Train: X={x_train.shape}, Y={y_train.shape}")
    print(f"  Val: X={x_val.shape}, Y={y_val.shape}")
    print(f"  Test: X={x_test.shape}, Y={y_test.shape}")
    
    # Save data
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"💾 Saving {cat} - X: {_x.shape}, Y: {_y.shape}")
        
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    
    print(f"✅ Data saved to {args.output_dir}")
    
    # Show time ranges if possible
    if hasattr(df, 'index'):
        total_time_steps = len(df)
        min_t = abs(min(x_offsets))
        max_t = abs(total_time_steps - abs(max(y_offsets)))
        
        print(f"\n⏰ Time ranges (approximately):")
        print(f"  Train: timesteps 0 to {num_train + min_t - 1}")
        print(f"  Val: timesteps {num_train + min_t} to {num_train + num_val + min_t - 1}")
        print(f"  Test: timesteps {num_train + num_val + min_t} to {max_t + min_t - 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, required=True, help="H5 file path.")
    parser.add_argument("--data_key", type=str, required=True, 
                       choices=['bike_pick', 'bike_drop', 'taxi_pick', 'taxi_drop'],
                       help="Data key to extract")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Input sequence length.")
    parser.add_argument("--seq_length_y", type=int, default=12, help="Output sequence length.")
    parser.add_argument("--y_start", type=int, default=1, help="Y prediction start offset.")
    parser.add_argument("--add_time_in_day", action='store_true', default=True)
    parser.add_argument("--add_day_in_week", action='store_true')

    args = parser.parse_args()
    
    print(f"🚀 Generating dataset with Y containing ONLY traffic data")
    print(f"📁 {args.data_key} from {args.traffic_df_filename} → {args.output_dir}")
    
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Overwrite? (y/n) ')).lower().strip()
        if reply[0] != 'y': 
            exit()
    else:
        os.makedirs(args.output_dir)
        
    generate_train_val_test(args)