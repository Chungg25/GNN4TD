from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import h5py


def generate_graph_seq2seq_io_data_combined(
        df_pick, df_drop, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False
):
    """
    Generate samples cho combined pick+drop data
    X: pick + drop + temporal features (3 channels cho X)
    Y: pick + drop (2 channels cho Y)
    """
    num_samples, num_nodes = df_pick.shape
    
    # 🚗 Traffic data cho X (pick và drop riêng biệt)
    traffic_pick = np.expand_dims(df_pick.values, axis=-1)  # (T, N, 1)
    traffic_drop = np.expand_dims(df_drop.values, axis=-1)  # (T, N, 1)
    
    # X sẽ có cả pick và drop làm input features
    x_feature_list = [traffic_pick, traffic_drop]  # 2 channels cho traffic
    
    # 🎯 Y data: pick và drop riêng biệt
    traffic_y_data = np.concatenate([traffic_pick, traffic_drop], axis=-1)  # (T, N, 2)
    
    print(f"🚗 Combined traffic data:")
    print(f"  Pick shape: {df_pick.values.shape}")
    print(f"  Drop shape: {df_drop.values.shape}")
    print(f"  X will have pick + drop + temporal features")
    print(f"  Y will have pick + drop: {traffic_y_data.shape}")
    
    # 📊 Add temporal features cho X
    if add_time_in_day:
        time_ind = (df_pick.index.values - df_pick.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        x_feature_list.append(time_in_day)
        
    if add_day_in_week:
        dow = df_pick.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        x_feature_list.append(dow_tiled)

    # X data: pick + drop + temporal features
    x_data = np.concatenate(x_feature_list, axis=-1)
    
    # Y data: pick + drop (không có temporal features)
    y_data = traffic_y_data  # Shape: (T, N, 2)
    
    print(f"📊 Final feature dimensions:")
    print(f"  X data shape: {x_data.shape} (pick + drop + temporal)")
    print(f"  Y data shape: {y_data.shape} (pick + drop only)")
    
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    
    for t in range(min_t, max_t):
        x.append(x_data[t + x_offsets, ...])
        y.append(y_data[t + y_offsets, ...])
        
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    
    print(f"🎯 Final sequence shapes:")
    print(f"  X: {x.shape} (samples, timesteps, nodes, features)")
    print(f"  Y: {y.shape} (samples, timesteps, nodes, 2)")
    
    return x, y


def load_h5_data_combined(filename, pick_key, drop_key):
    """Load both pick and drop data from H5 file"""
    print(f"🔍 Loading combined data from {filename}")
    print(f"📥 Pick key: {pick_key}, Drop key: {drop_key}")
    
    with h5py.File(filename, 'r') as f:
        print(f"📋 Available keys in file: {list(f.keys())}")
        
        if pick_key not in f.keys():
            raise ValueError(f"Pick key '{pick_key}' not found in file. Available: {list(f.keys())}")
        if drop_key not in f.keys():
            raise ValueError(f"Drop key '{drop_key}' not found in file. Available: {list(f.keys())}")
        
        pick_data = np.array(f[pick_key])
        drop_data = np.array(f[drop_key])
        
        print(f"✅ Pick data loaded: shape = {pick_data.shape}")
        print(f"✅ Drop data loaded: shape = {drop_data.shape}")
    
    # Verify shapes match
    if pick_data.shape != drop_data.shape:
        raise ValueError(f"Pick and drop data shapes don't match: {pick_data.shape} vs {drop_data.shape}")
    
    T, N = pick_data.shape
    
    # Create time index
    if 'NYC' in filename:
        start_time = '2016-04-01 00:00:00'
    else:
        start_time = '2024-07-01 00:00:00'
    
    time_index = pd.date_range(start=start_time, periods=T, freq='30min')
    
    # Create DataFrames
    df_pick = pd.DataFrame(pick_data, index=time_index, columns=[f'node_{i}' for i in range(N)])
    df_drop = pd.DataFrame(drop_data, index=time_index, columns=[f'node_{i}' for i in range(N)])
    
    print(f"📊 DataFrames created:")
    print(f"  Pick: {df_pick.shape}")
    print(f"  Drop: {df_drop.shape}")
    print(f"⏰ Time range: {df_pick.index.min()} to {df_pick.index.max()}")
    
    return df_pick, df_drop


def generate_train_val_test_combined(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    
    # Load both pick and drop data
    df_pick, df_drop = load_h5_data_combined(
        args.traffic_df_filename, 
        args.pick_key, 
        args.drop_key
    )
    
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + args.y_start), 1))
    
    print(f"🔧 X offsets: {x_offsets}")
    print(f"🎯 Y offsets: {y_offsets}")
    
    # Generate combined sequences
    x, y = generate_graph_seq2seq_io_data_combined(
        df_pick=df_pick,
        df_drop=df_drop,
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
    
    # Data statistics
    print(f"\n📊 Data Statistics:")
    print(f"  X input features: {x.shape[-1]} (pick + drop + temporal)")
    print(f"  Y output channels: {y.shape[-1]} (pick + drop)")
    print(f"  Pick traffic range: [{df_pick.values.min():.2f}, {df_pick.values.max():.2f}]")
    print(f"  Drop traffic range: [{df_drop.values.min():.2f}, {df_drop.values.max():.2f}]")
    
    # Calculate feature breakdown
    n_temporal_features = x.shape[-1] - 2  # Total features - pick - drop
    print(f"  Input feature breakdown:")
    print(f"    - Pick traffic: 1 channel")
    print(f"    - Drop traffic: 1 channel") 
    print(f"    - Temporal features: {n_temporal_features} channels")
    
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
    
    print(f"✅ Combined data saved to {args.output_dir}")
    
    # Save metadata
    metadata = {
        'data_type': 'combined_pick_drop_separate_channels',
        'pick_key': args.pick_key,
        'drop_key': args.drop_key,
        'input_features': int(x.shape[-1]),
        'output_channels': int(y.shape[-1]),
        'num_nodes': int(x.shape[2]),
        'seq_length_x': seq_length_x,
        'seq_length_y': seq_length_y,
        'feature_breakdown': {
            'pick_traffic': 1,
            'drop_traffic': 1,
            'temporal_features': int(x.shape[-1] - 2)
        },
        'dataset_splits': {
            'train': int(num_train),
            'val': int(num_val),
            'test': int(num_test),
            'total': int(num_samples)
        },
        'time_range': {
            'start': str(df_pick.index.min()),
            'end': str(df_pick.index.max()),
            'frequency': '30min'
        }
    }
    
    import json
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Metadata saved to {args.output_dir}/metadata.json")
    
    # Show time ranges
    if hasattr(df_pick, 'index'):
        total_time_steps = len(df_pick)
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
    parser.add_argument("--pick_key", type=str, required=True, 
                       choices=['bike_pick', 'taxi_pick'],
                       help="Pick data key to extract")
    parser.add_argument("--drop_key", type=str, required=True,
                       choices=['bike_drop', 'taxi_drop'], 
                       help="Drop data key to extract")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Input sequence length.")
    parser.add_argument("--seq_length_y", type=int, default=12, help="Output sequence length.")
    parser.add_argument("--y_start", type=int, default=1, help="Y prediction start offset.")
    parser.add_argument("--add_time_in_day", action='store_true', default=True)
    parser.add_argument("--add_day_in_week", action='store_true')

    args = parser.parse_args()
    
    # Validate pick and drop keys match
    if ('bike' in args.pick_key) != ('bike' in args.drop_key):
        raise ValueError("Pick and drop keys must be for the same vehicle type (both bike or both taxi)")
    
    vehicle_type = 'bike' if 'bike' in args.pick_key else 'taxi'
    
    print(f"🚀 Generating COMBINED {vehicle_type} dataset with SEPARATE CHANNELS")
    print(f"📁 {args.pick_key} + {args.drop_key} from {args.traffic_df_filename} → {args.output_dir}")
    print(f"🎯 X: Pick + Drop + temporal features (separate channels)")
    print(f"🎯 Y: Pick + Drop predictions (separate channels)")
    
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Overwrite? (y/n) ')).lower().strip()
        if reply[0] != 'y': 
            exit()
    else:
        os.makedirs(args.output_dir)
        
    generate_train_val_test_combined(args)