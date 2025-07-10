import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from model import STIDGCN
from ranger21 import Ranger
import torch.optim as optim
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="number of input_dim")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=300, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    help="save path",
)
parser.add_argument("--expid", type=int, default=1, help="experiment id")
parser.add_argument(
    "--es_patience",
    type=int,
    default=30,
    help="quit if no improvement after this many iterations",
)

args = parser.parse_args()


def calculate_pcc(y_true, y_pred):
    """
    Tính Pearson Correlation Coefficient
    """
    y_true_flat = y_true.detach().cpu().numpy().flatten()
    y_pred_flat = y_pred.detach().cpu().numpy().flatten()
    
    # Loại bỏ NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    correlation, _ = pearsonr(y_true_clean, y_pred_clean)
    return correlation if not np.isnan(correlation) else 0.0

def calculate_pcc_per_feature(y_true, y_pred, feature_names=['pick', 'drop']):
    pcc_results = {}
    
    for i, feature_name in enumerate(feature_names):
        y_true_feature = y_true[..., i].detach().cpu().numpy().flatten()
        y_pred_feature = y_pred[..., i].detach().cpu().numpy().flatten()
        
        mask = ~(np.isnan(y_true_feature) | np.isnan(y_pred_feature))
        y_true_clean = y_true_feature[mask]
        y_pred_clean = y_pred_feature[mask]
        
        if len(y_true_clean) == 0:
            pcc_results[feature_name] = 0.0
        else:
            correlation, _ = pearsonr(y_true_clean, y_pred_clean)
            pcc_results[feature_name] = correlation if not np.isnan(correlation) else 0.0
    
    return pcc_results


class trainer:
    def __init__(
        self,
        scaler,
        input_dim,
        num_nodes,
        channels,
        dropout,
        lrate,
        wdecay,
        device,
        granularity,
    ):
        self.model = STIDGCN(
            device, input_dim, num_nodes, channels, granularity, dropout
        )
        self.model.to(device)
        # self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        print("The number of parameters: {}".format(self.model.param_num()))
        print(self.model)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output[:, :, -1, :, :]
        predict = self.scaler.inverse_transform(output.cpu().numpy())
        predict = torch.from_numpy(predict).to(input.device)
        # output = output.transpose(1, 3)
        # real = torch.unsqueeze(real_val, dim=1)
        # predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.MAPE_torch(predict, real_val, 0.0).item()
        rmse = util.RMSE_torch(predict, real_val, 0.0).item()
        wmape = util.WMAPE_torch(predict, real_val, 0.0).item()
        pcc = calculate_pcc(real_val, predict)
        return loss.item(), mape, rmse, wmape, pcc

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            # output = output.transpose(1, 3)
            output = output[:, :, -1, :, :]
            # real = torch.unsqueeze(real_val, dim=1)
            # predict = self.scaler.inverse_transform(output)
            predict = self.scaler.inverse_transform(output.cpu().numpy())
            predict = torch.from_numpy(predict).to(input.device)
            loss = self.loss(predict, real_val, 0.0)
            mape = util.MAPE_torch(predict, real_val, 0.0).item()
            rmse = util.RMSE_torch(predict, real_val, 0.0).item()
            wmape = util.WMAPE_torch(predict, real_val, 0.0).item()
            pcc = calculate_pcc(real_val, predict)
            return loss.item(), mape, rmse, wmape, pcc


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main():
    seed_it(6666)

    data = args.data

    if args.data == "PEMS08":
        args.data = "data//" + args.data
        num_nodes = 170
        granularity = 288
        channels = 48

    elif args.data == "NYC/bike_combined":  
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        num_nodes = 358
        args.epochs = 300
        args.es_patience = 100
        granularity = 288
        channels = 32

    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        num_nodes = 307
        granularity = 288
        channels = 48

    elif args.data == "PEMS07":
        args.data = "data//" + args.data
        num_nodes = 883
        granularity = 288
        channels = 128

    elif args.data == "bike_drop":
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32

    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32
    
    elif args.data == "NYC/bike_pick":
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32
    
    elif args.data == "NYC/bike_drop":
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32

    elif args.data == "NYC/taxi_drop":
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96
    
    elif args.data == "NYC/taxi_pick":
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96
    
    elif args.data == "DC/taxi_pick":
        args.data = "data//" + args.data
        num_nodes = 271
        granularity = 48
        channels = 96
    
    elif args.data == "DC/taxi_drop":
        args.data = "data//" + args.data
        num_nodes = 271
        granularity = 48
        channels = 96
    
    elif args.data == "CHI/taxi_drop":
        args.data = "data//" + args.data
        num_nodes = 77
        granularity = 48
        channels = 96
    
    elif args.data == "CHI/taxi_pick":
        args.data = "data//" + args.data
        num_nodes = 77
        granularity = 48
        channels = 96
    
    elif args.data == "DC/bike_drop":
        args.data = "data//" + args.data
        num_nodes = 196
        granularity = 48
        channels = 32
    
    elif args.data == "DC/bike_pick":
        args.data = "data//" + args.data
        num_nodes = 196
        granularity = 48
        channels = 32
    
    elif args.data == "BOSTON/bike_pick":
        args.data = "data//" + args.data
        num_nodes = 145
        granularity = 48
        channels = 32

    elif args.data == "BOSTON/bike_drop":
        args.data = "data//" + args.data
        num_nodes = 145
        granularity = 48
        channels = 32
    
    elif args.data == "BAY/bike_pick":
        args.data = "data//" + args.data
        num_nodes = 208
        granularity = 48
        channels = 32
    
    elif args.data == "BAY/bike_drop":
        args.data = "data//" + args.data
        num_nodes = 208
        granularity = 48
        channels = 32

    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96

    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96

    device = torch.device(args.device)

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    path = args.save + data + "/"

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(
        scaler,
        args.input_dim,
        num_nodes,
        channels,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        granularity,
    )

    print("start training...", flush=True)

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []
        train_pcc = []

        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            # trainx = torch.Tensor(x).to(device)
            # trainx = trainx.transpose(1, 3)
            # trainy = torch.Tensor(y).to(device)
            # trainy = trainy.transpose(1, 3)
            # metrics = engine.train(trainx, trainy[:, 0, :, :])
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)

            if iter == 0:
                print(f"🔍 Raw data shapes:")
                print(f"  x: {x.shape}")
                print(f"  y: {y.shape}")
                print(f"🔍 After torch.Tensor:")
                print(f"  trainx: {trainx.shape}")
                print(f"  trainy: {trainy.shape}")
                print(f"🔍 trainx features (last dim): {trainx.shape[-1]}")
                print(f"🔍 Expected input_dim: {args.input_dim}")
                
                # 🔧 KIỂM TRA NẾU DATA CÓ ĐỦ FEATURES
                if trainx.shape[-1] != args.input_dim:
                    print(f"❌ ERROR: Data has {trainx.shape[-1]} features but model expects {args.input_dim}")
                    print(f"📊 Adjusting input_dim to match data...")
                    args.input_dim = trainx.shape[-1]
                    
                    # 🔧 TẠO LẠI MODEL VỚI ĐÚNG input_dim
                    print(f"🔄 Recreating model with input_dim={args.input_dim}")
                    engine = trainer(
                        scaler,
                        args.input_dim,
                        num_nodes,
                        channels,
                        args.dropout,
                        args.learning_rate,
                        args.weight_decay,
                        device,
                        granularity,
                    )
  
            trainx = trainx.permute(0, 3, 2, 1)  
            trainy = trainy.permute(0, 2, 1, 3)

            if iter == 0:
                print(f"🔍 After permutation:")
                print(f"  trainx: {trainx.shape}")
                print(f"  trainy: {trainy.shape}")
                print(f"🔍 Model expects input: [batch, {args.input_dim}, {num_nodes}, time]")
                print(f"🔍 Model expects output: [batch, {num_nodes}, time, 12, 2]")
            
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])
            train_pcc.append(metrics[4])

            if iter % args.print_every == 0:
                log = "Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, Train PCC: {:.4f}"
                print(
                    log.format(
                        iter,
                        train_loss[-1],
                        train_rmse[-1],
                        train_mape[-1],
                        train_wmape[-1],
                        train_pcc[-1],
                    ),
                    flush=True,
                )
        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []
        valid_pcc = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device)
            # testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            # testy = testy.transpose(1, 3)
            # metrics = engine.eval(testx, testy[:, 0, :, :])
            testx = testx.permute(0, 3, 2, 1)
            testy = testy.permute(0, 2, 1, 3)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])
            valid_pcc.append(metrics[4])

        s2 = time.time()

        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_pcc = np.mean(train_pcc)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_pcc = np.mean(valid_pcc)

        his_loss.append(mvalid_loss)
        train_m = dict(
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            train_pcc=np.mean(train_pcc),
            valid_loss=np.mean(valid_loss),
            valid_rmse=np.mean(valid_rmse),
            valid_mape=np.mean(valid_mape),
            valid_wmape=np.mean(valid_wmape),
            valid_pcc=np.mean(valid_pcc),
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, Train PCC: {:.4f}"
        print(
            log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape, mtrain_pcc),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}, Valid PCC: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape, mvalid_pcc),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            if i < 100:
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:", mvalid_loss, end=", ")
                print("epoch: ", i)

            elif i > 100:
                outputs = []
                realy = torch.Tensor(dataloader["y_test"]).to(device)
                # realy = realy.transpose(1, 3)[:, 0, :, :]

                for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    # testx = testx.transpose(1, 3)
                    testx = testx.permute(0, 3, 2, 1)
                    with torch.no_grad():
                        preds = engine.model(testx)
                        preds = preds[:, :, -1, :, :]
                    # outputs.append(preds.squeeze())
                    outputs.append(preds)

                yhat = torch.cat(outputs, dim=0)
                yhat = yhat[: realy.size(0), ...]

                amae = []
                amape = []
                awmape = []
                armse = []
                apcc = []
                test_m = []

                for j in range(12):
                    # pred = scaler.inverse_transform(yhat[:, :, j])
                    pred_step = scaler.inverse_transform(yhat[:, :, j, :].cpu().numpy())
                    # real = realy[:, :, j]
                    real_step = realy[:, :, j, :].cpu().numpy()
                    # metrics = util.metric(pred, real)
                    # pcc = calculate_pcc(real, pred)
                    pred_step = torch.from_numpy(pred_step).to(device)
                    real_step = torch.from_numpy(real_step).to(device)
                    metrics = util.metric(pred_step, real_step)
                    pcc = calculate_pcc(real_step, pred_step)
                    log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}, Test PCC: {:.4f}"
                    print(
                        log.format(
                            j + 1, metrics[0], metrics[2], metrics[1], metrics[3], pcc
                        )
                    )

                    test_m = dict(
                        test_loss=np.mean(metrics[0]),
                        test_rmse=np.mean(metrics[2]),
                        test_mape=np.mean(metrics[1]),
                        test_wmape=np.mean(metrics[3]),
                        test_pcc=pcc,
                    )
                    test_m = pd.Series(test_m)

                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])
                    awmape.append(metrics[3])
                    apcc.append(pcc)

                log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}, Test PCC: {:.4f}"
                print(
                    log.format(
                        np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape), np.mean(apcc)
                    )
                )

                if np.mean(amae) < test_log:
                    test_log = np.mean(amae)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mae = 0
                    print("Test low! Updating! Test Loss:", np.mean(amae), end=", ")
                    print("Test low! Updating! Valid Loss:", mvalid_loss, end=", ")
                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mae += 1
                    print("No update")

        else:
            epochs_since_best_mae += 1
            print("No update")

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")
        if epochs_since_best_mae >= args.es_patience and i >= 300:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    # realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        # testx = testx.transpose(1, 3)
        testx = testx.permute(0, 3, 2, 1)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds[:, :, -1, :, :]
            # preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    yhat_denorm = scaler.inverse_transform(yhat.cpu().numpy())
    yhat_denorm = torch.from_numpy(yhat_denorm).to(device)

    results = util.comprehensive_evaluation(yhat_denorm, realy)
    util.print_evaluation_summary(results, "Final Test Results")

    amae = []
    amape = []
    armse = []
    awmape = []
    apcc = []

    test_m = []

    for i in range(12):
        # pred = scaler.inverse_transform(yhat[:, :, i])
        # real = realy[:, :, i]
        pred_step = yhat_denorm[:, :, i, :]
        real_step = realy[:, :, i, :]
        # metrics = util.metric(pred, real)
        # pcc = calculate_pcc(real, pred)
        metrics = util.metric(pred_step, real_step)
        pcc = calculate_pcc(real_step, pred_step)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}, Test PCC: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[2], metrics[1], metrics[3], pcc))

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
            test_pcc=pcc,
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])
        apcc.append(pcc)

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}, Test PCC: {:.4f}"
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape), np.mean(apcc)))

    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
        test_pcc=np.mean(apcc),
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{path}/test.csv")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))