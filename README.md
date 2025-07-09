# GNN4TD
GNN4TD là dự án khảo sát các mô hình nổi bật trong bài toán Dự đoán nhu cầu vận chuyển hành khách. Bên cạnh hai tập dữ liệu nổi tiếng trong lĩnh vực là NYC_Taxi và NYC_Bike, chúng tôi cung cấp thêm ba tập dữ liệu mới là dữ liệu về nhu cầu thuê xe đạp tại ba thành phố lần lượt là Boston, Bay Area và Washington DC. 

Các mô hình thực nghiệm:

- **CCRNN**: Coupled Layer-wise Recurrent Neural Network
- **MSDR**: Multi-Step Dependency Relation Networks
- **MVFN**: Multi-View Fusion Neural Network
- **DMGL**: Dynamic and Multi-Scale Graph Learning

Các tập dữ liệu sử dụng

- **NYC_Taxi**: Dữ liệu về nhu cầu đặt xe taxi tại thành phố New York, USA (01/04/2016 - 30/06/2016)
- **NYC_Bike**: Dữ liệu về nhu cầu thuê xe đạp tại thành phố New York, USA (01/04/2016 - 30/06/2016)
- **BOS_Bike**: Dữ liệu về nhu cầu thuê xe đạp tại Boston, USA (01/07/2024 - 30/09/2024)
- **BAY_Bike**: Dữ liệu về nhu cầu thuê xe đạp tại Bay Area, USA (01/07/2024 - 30/09/2024) 
- **DC_Bike**: Dữ liệu về nhu cầu thuê xe đạp tại Washington DC, USA (01/07/2024 - 30/09/2024)  

## Các chạy thực nghiệm các mô hình

### 1. CCRNN

```python
!python main.py --type bike --model CCRNN --city BAY
!python main.py --type bike --model CCRNN --city DC
!python main.py --type bike --model CCRNN --city BOSTON
!python main.py --type bike --model CCRNN --city NYC
!python main.py --type taxi --model CCRNN --city NYC
```

### 2. MSDR
```python
!python main.py --type bike --model GMSDR --city BAY
!python main.py --type bike --model GMSDR --city DC
!python main.py --type bike --model GMSDR --city BOSTON
!python main.py --type bike --model GMSDR --city NYC
!python main.py --type taxi --model GMSDR --city NYC
```

### 3. MVFN
Truy cập thư mục MVFN
```python
cd MVFN
```
```python
!python train.py --drop data/BOSTON/BIKE/bike_drop.csv --pick data/BOSTON/BIKE/bike_pick.csv --adj_data data/BOSTON/BIKE/dis_bb.csv --parameter parameter/bike
!python train.py --drop data/DC/BIKE/bike_drop.csv --pick data/DC/BIKE/bike_pick.csv --adj_data data/DC/BIKE/dis_bb.csv --parameter parameter/bike
!python train.py --drop data/BAY/BIKE/bike_drop.csv --pick data/BAY/BIKE/bike_pick.csv --adj_data data/BAY/BIKE/dis_bb.csv --parameter parameter/bike
!python train.py --drop data/bike_drop.csv --pick data/bike_pick.csv --adj_data data/dis_bb.csv --parameter parameter/bike
!python train.py --drop data/taxi_drop.csv --pick data/taxi_pick.csv --adj_data data/dis_tt.csv --parameter parameter/taxi
```

### 4. DMGL
Truy cập thư mục DMGL
```python
cd DMGL
```
```python
! python train_multi_step.py --data nyc-taxi --num_nodes 266 --runs 1 --fc_dim 95744
! python train_multi_step.py --data nyc-bike --num_nodes 250 --runs 1 --fc_dim 95744
! python train_multi_step.py --data bos-bike --num_nodes 201 --runs 1 --fc_dim 97280
! python train_multi_step.py --data bay-bike --num_nodes 133 --runs 1 --fc_dim 97280
! python train_multi_step.py --data dc-bike --num_nodes 117 --runs 1 --fc_dim 97280
```
# GNN4TD
GNN4TD là dự án dự đoán lưu lượng giao thông dựa trên dữ liệu thời gian thực từ nhiều nguồn khác nhau. Dữ liệu được lấy từ các tập sau:

- **NYC_bike**: Dữ liệu xe đạp tại New York City  
- **NYC_taxi**: Dữ liệu taxi tại New York City  
- **BOSTON_BIKE**: Dữ liệu xe đạp tại Boston  
- **BAY_BIKE**: Dữ liệu xe đạp tại khu vực Bay Area  
- **DC_Bike**: Dữ liệu xe đạp tại Washington DC  

## Các mô hình sử dụng

### 1. CCRNN

```python
!python main.py --type bike --model CCRNN --city BAY
!python main.py --type bike --model CCRNN --city DC
!python main.py --type bike --model CCRNN --city BOSTON
!python main.py --type bike --model CCRNN --city NYC
!python main.py --type taxi --model CCRNN --city NYC
```

### 2. MSDR
```python
!python main.py --type bike --model GMSDR --city BAY
!python main.py --type bike --model GMSDR --city DC
!python main.py --type bike --model GMSDR --city BOSTON
!python main.py --type bike --model GMSDR --city NYC
!python main.py --type taxi --model GMSDR --city NYC
```

### 3. MVFN
Trước khi chạy phải vào thư mục MVFN
```python
cd MVFN
```
```python
!python train.py --drop data/BOSTON/BIKE/bike_drop.csv --pick data/BOSTON/BIKE/bike_pick.csv --adj_data data/BOSTON/BIKE/dis_bb.csv --parameter parameter/bike
!python train.py --drop data/DC/BIKE/bike_drop.csv --pick data/DC/BIKE/bike_pick.csv --adj_data data/DC/BIKE/dis_bb.csv --parameter parameter/bike
!python train.py --drop data/BAY/BIKE/bike_drop.csv --pick data/BAY/BIKE/bike_pick.csv --adj_data data/BAY/BIKE/dis_bb.csv --parameter parameter/bike
!python train.py --drop data/bike_drop.csv --pick data/bike_pick.csv --adj_data data/dis_bb.csv --parameter parameter/bike
!python train.py --drop data/taxi_drop.csv --pick data/taxi_pick.csv --adj_data data/dis_tt.csv --parameter parameter/taxi
```

### 4. DMGL
Trước khi chạy phải vào thư mục DMGL
```python
cd DMGL
```
```python
! python train_multi_step.py --data nyc-taxi --num_nodes 266 --runs 1 --fc_dim 95744
! python train_multi_step.py --data nyc-bike --num_nodes 250 --runs 1 --fc_dim 95744
! python train_multi_step.py --data bos-bike --num_nodes 201 --runs 1 --fc_dim 97280
! python train_multi_step.py --data bay-bike --num_nodes 133 --runs 1 --fc_dim 97280
! python train_multi_step.py --data dc-bike --num_nodes 117 --runs 1 --fc_dim 97280
```