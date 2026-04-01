import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path: sys.path.append(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics_dl import metric
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.plot_results import predict_result_visual
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.model_memory import model_memory_size
from utils.timefeatures import time_features
from utils.log_util import logger

import warnings
warnings.filterwarnings('ignore')


class Exp_Zero_Shot_Forecast(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Exp_Zero_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        # 时间序列模型初始化
        logger.info(f"Initializing model {self.args.model}...")
        model = self.model_dict[self.args.model](self.args).float()
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # 打印模型参数量
        model_memory_size(model, verbose=True)
        
        return model

    def _get_data(self, flag):
        """
        数据集构建
        """
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_criterion(self):
        """
        评价指标
        """
        if self.args.loss == "MSE":
            return nn.MSELoss()
        elif self.args.loss == "MAPE":
            return mape_loss()
        elif self.args.loss == "MASE":
            return mase_loss()
        elif self.args.loss == "SMAPE":
            return smape_loss()
        elif self.args.loss == "L1":
            return nn.L1Loss()
    
    def _select_optimizer(self):
        """
        优化器
        """
        if self.args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        
        return optimizer
    
    def _get_model_path(self, setting):
        """
        模型保存路径，如果进行模型训练任务，则需要保存模型
        """
        # 模型保存路径
        model_path = Path(self.args.checkpoints).joinpath(setting)
        model_path.mkdir(parents=True, exist_ok=True)
        # 最优模型保存路径
        model_checkpoint_path = model_path.joinpath("checkpoint.pth")
        
        return model_checkpoint_path

    def _get_test_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = Path(self.args.test_results).joinpath(setting)
        results_path.mkdir(parents=True, exist_ok=True)
        
        return results_path

    def _get_predict_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = Path(self.args.forecast_results).joinpath(setting)
        results_path.mkdir(parents=True, exist_ok=True)
        
        return results_path

    def _test_results_save(self, preds, trues, setting, path):
        """
        测试结果保存
        """
        # ------------------------------
        # 计算测试结果评价指标
        # ------------------------------
        (r2, mse, rmse, mae, mape, mape_accuracy, mspe, dtw) = metric(
            preds, trues, 
            use_dtw=self.args.use_dtw
        )
        # summary_line = f"Test results: r2:{r2:.4f}, mse:{mse:.4f}, rmse:{rmse:.4f}, mae:{mae:.4f} mape:{mape:.4f}, mape accuracy:{mape_accuracy:.4f}, mspe:{mspe:.4f}, dtw:{dtw:.4f}"
        summary_line = f"Test results: r2:{r2:.4f}, mse:{mse:.4f}, rmse:{rmse:.4f}, mae:{mae:.4f} mape:{mape:.4f}, mape accuracy:{mape_accuracy:.4f}, mspe:{mspe:.4f}"
        logger.info(summary_line)
        with open(Path(path).joinpath("result_forecast.txt"), 'a', encoding='utf-8') as file:
            file.write(setting + "  \n")
            file.write(summary_line)
            file.write('\n')
            file.write('\n')
            file.close()
        # ------------------------------
        # 测试集上的预测值、真实值 
        # ------------------------------
        test_results = pd.DataFrame({
            "preds": preds.reshape(1, -1)[0], 
            "trues": trues.reshape(1, -1)[0]
        }, index=range(len(preds.reshape(1, -1)[0])))
        test_results.to_csv(Path(path).joinpath("test_results.csv"), index=False, encoding="utf-8")
        logger.info(f"test_results: \n{test_results}")
        # np.save(Path(path).joinpath('metrics.npy'), np.array([r2, mae, mse, rmse, mape, mape_accuracy, mspe, dtw]))
        # np.save(Path(path).joinpath('preds.npy'), preds)
        # np.save(Path(path).joinpath('trues.npy'), trues)
    
    def _pred_results_save(self, trues_df, preds_df, preds=None, path="./", setting=None):
        """
        预测结果保存
        """
        if preds is not None:
            np.save(Path(path).joinpath("prediction.npy"), preds) 

        if trues_df is not None:
            trues_df.to_csv(path.joinpath('history.csv'), index=False, encoding="utf_8_sig")
        
        if preds_df is not None:
            preds_df.to_csv(path.joinpath('forecast.csv'), index=False, encoding="utf_8_sig")
        
        with open(os.path.join(path, 'summary.txt'), 'w', encoding='utf-8') as summary_file:
            summary_file.write(setting + '\n')
            summary_file.write(f'prediction only: no ground truth available\n')
            summary_file.write(f'history_points:{len(trues_df)}, forecast_points:{len(preds_df)}\n')
            summary_file.write(f'forecast_target:{preds_df.columns[-1]}\n')
    
    def test(self, setting, test=0):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag='test')
        # 测试结果保存地址
        logger.info(f"{40 * '-'}")
        logger.info(f"Test results will be saved in path:")
        logger.info(f"{40 * '-'}")
        test_results_path = self._get_test_results_path(setting) 
        logger.info(test_results_path)
        # 模型开始测试
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start testing...")
        logger.info(f"{40 * '-'}")
        # 模型测试次数
        test_steps = len(test_loader)
        logger.info(f"Test total steps: {test_steps}")
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        preds, trues = [], []
        preds_flat, trues_flat = [], [] 
        with torch.no_grad():
            for iters, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                logger.info(f"Test step: {iters} running...")
                
                # 前向传播
                # 数据预处理
                # ---------------------
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                # ---------------------
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # ---------------------
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # pred and true process
                # ---------------------
                # pred and true 提取
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                # output detach device
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # 输入输出逆转换
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                # 预测值/真实值提取
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                # 测试结果收集
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)

                # 预测数据可视化
                if iters % 10 == 0:
                    inputs = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inputs.shape
                        inputs = test_data.inverse_transform(inputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    true_plot = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    pred_plot = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    predict_result_visual(pred_plot, true_plot, test_results_path, iters=iters)
        # 测试结果处理
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        logger.info(f'test preds shape: {preds.shape}, trues shape: {trues.shape}')
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        logger.info(f'test preds shape: {preds.shape} tures shape: {trues.shape}')
        # 测试结果收集
        logger.info(f"{40 * '-'}")
        logger.info(f"Test metric results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._test_results_save(preds.reshape(-1, 1), trues.reshape(-1, 1), setting, test_results_path)
        logger.info(test_results_path)
        # 测试结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Test visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        if self.args.features == 'M':
            preds_flat = np.concatenate(preds, axis = 0)[:, -1]
            trues_flat = np.concatenate(trues, axis = 0)[:, -1]
        else:
            preds_flat = np.concatenate(preds, axis = 0)
            trues_flat = np.concatenate(trues, axis = 0)
        predict_result_visual(preds_flat, trues_flat, path=test_results_path, iters=None) 
        logger.info(test_results_path)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return

    def forecast(self, setting):
        """
        模型预测（推理）
        """
        # 构建预测数据集
        (batch_x, batch_x_mark, 
         dec_inp, batch_y_mark, 
         history_dates, future_dates, feature_names) = self._build_predict_inputs()
        # 模型预测结果保存地址
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast results will be saved in path:")
        logger.info(f"{40 * '-'}")
        pred_results_path = self._get_predict_results_path(setting)
        logger.info(pred_results_path)
        # 模型开始预测
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start forecasting...")
        logger.info(f"{40 * '-'}")
        # 模型评估模式
        self.model.eval()
        with torch.no_grad():
            if self.args.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 预测结果提取
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, :]
        outputs = outputs[:, :, f_dim:]
        preds = outputs.detach().cpu().numpy()[0]
        
        # 预测结果保存
        pred_columns = feature_names[f_dim:] if f_dim != 0 else feature_names
        if len(pred_columns) != preds.shape[-1]:
            pred_columns = pred_columns[-preds.shape[-1]:]
        # 历史数据表
        history_frame = pd.DataFrame(batch_x.detach().cpu().numpy()[0], columns=feature_names)
        # history_frame.insert(0, 'date', history_dates.astype(str))
        history_frame.insert(0, 'date', history_dates)
        # 预测数据表
        forecast_frame = pd.DataFrame(preds, columns=pred_columns)
        # forecast_frame.insert(0, 'date', future_dates.astype(str))
        forecast_frame.insert(0, 'date', future_dates)
        # 最终预测值保存
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._pred_results_save(history_frame, forecast_frame,  preds, pred_results_path, setting)
        logger.info(pred_results_path)
        # 预测结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        history_target = history_frame[pred_columns[-1]].to_numpy()
        forecast_target = forecast_frame[pred_columns[-1]].to_numpy()
        forecast_target = np.concatenate((history_target, forecast_target), axis=0)
        predict_result_visual(forecast_target, history_target, pred_results_path, iters=None)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecasting Finished!")
        logger.info(f"{40 * '-'}")
        
        return

    def _build_predict_inputs(self):
        # data read
        # ------------------------------
        file_path = os.path.join(self.args.root_path, self.args.data_path)
        df_raw = pd.read_csv(file_path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        # data reorder
        # ------------------------------
        cols = list(df_raw.columns)
        cols.remove('date')
        if self.args.target in cols:
            cols.remove(self.args.target)
        ordered_cols = ['date'] + cols + [self.args.target]
        df_raw = df_raw[ordered_cols]
        # data split
        # ------------------------------
        if self.args.features in ['M', 'MS']:
            feature_cols = df_raw.columns[1:]
            df_data = df_raw[feature_cols]
        else:
            feature_cols = [self.args.target]
            df_data = df_raw[[self.args.target]]

        if len(df_data) < self.args.seq_len:
            raise ValueError(f'not enough rows for prediction: need at least seq_len={self.args.seq_len}, got {len(df_data)}')
        # ------------------------------
        # history data
        history_values = df_data.iloc[-self.args.seq_len:].values.astype(np.float32)
        history_dates = df_raw['date'].iloc[-self.args.seq_len:].to_numpy()

        # future features
        future_dates = pd.date_range(start=df_raw['date'].iloc[-1], periods=self.args.pred_len + 1, freq=self.args.freq)[1:].to_numpy()
        label_len = max(int(self.args.label_len), 0)
        label_dates = pd.to_datetime(history_dates[-label_len:]) if label_len > 0 else pd.to_datetime([])
        decoder_dates = np.concatenate([label_dates.to_numpy(), future_dates], axis=0) if label_len > 0 else future_dates

        # time features
        # ------------------------------
        if self.args.embed != 'timeF':
            history_stamp = self._build_calendar_features(pd.to_datetime(history_dates))
            decoder_stamp = self._build_calendar_features(pd.to_datetime(decoder_dates))
        else:
            history_stamp = time_features(pd.to_datetime(history_dates), freq=self.args.freq).transpose(1, 0)
            decoder_stamp = time_features(pd.to_datetime(decoder_dates), freq=self.args.freq).transpose(1, 0)
        
        # model inputs
        # ------------------------------
        batch_x = torch.from_numpy(history_values).unsqueeze(0).float().to(self.device)
        batch_x_mark = torch.from_numpy(history_stamp.astype(np.float32)).unsqueeze(0).float().to(self.device)
        batch_y_mark = torch.from_numpy(decoder_stamp.astype(np.float32)).unsqueeze(0).float().to(self.device)

        if label_len > 0:
            label_context = history_values[-label_len:]
        else:
            label_context = np.zeros((0, history_values.shape[1]), dtype=np.float32)
        future_zeros = np.zeros((self.args.pred_len, history_values.shape[1]), dtype=np.float32)
        dec_inp = np.concatenate([label_context, future_zeros], axis=0)
        dec_inp = torch.from_numpy(dec_inp).unsqueeze(0).float().to(self.device)

        return (batch_x, batch_x_mark, dec_inp, batch_y_mark, history_dates, future_dates, list(feature_cols))

    @staticmethod
    def _build_calendar_features(dates):
        df_stamp = pd.DataFrame({'date': pd.to_datetime(dates)})
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
        if (df_stamp.date.dt.minute != 0).any():
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
        return df_stamp.drop(columns=['date']).values
