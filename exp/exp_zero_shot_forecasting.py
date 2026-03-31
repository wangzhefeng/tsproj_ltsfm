import os
os.environ.setdefault('MPLCONFIGDIR', os.path.abspath('.matplotlib'))

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import time
import warnings
import numpy as np
import pandas as pd
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class Exp_Zero_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Zero_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if hasattr(model, 'task_name') and model.task_name == 'zero_shot_predict':
            model.task_name = 'zero_shot_forecast'

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        output_dir = os.path.join('./results', setting)
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # start_time = time.time()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # print("Test cost time: {}".format(time.time() - start_time))
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(output_dir, f'{i}.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        summary_line = 'mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw)
        print(summary_line)
        with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as summary_file:
            summary_file.write(setting + '\n')
            summary_file.write(summary_line + '\n')

        np.save(os.path.join(output_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(output_dir, 'pred.npy'), preds)
        np.save(os.path.join(output_dir, 'true.npy'), trues)

        return

    def predict(self, setting):
        output_dir = os.path.join('./results', setting)
        os.makedirs(output_dir, exist_ok=True)

        batch_x, batch_x_mark, dec_inp, batch_y_mark, history_dates, future_dates, feature_names = self._build_predict_inputs()

        self.model.eval()
        with torch.no_grad():
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, :]
        outputs = outputs[:, :, f_dim:]
        preds = outputs.detach().cpu().numpy()[0]

        pred_columns = feature_names[f_dim:] if f_dim != 0 else feature_names
        if len(pred_columns) != preds.shape[-1]:
            pred_columns = pred_columns[-preds.shape[-1]:]

        history_frame = pd.DataFrame(
            batch_x.detach().cpu().numpy()[0],
            columns=feature_names,
        )
        history_frame.insert(0, 'date', history_dates.astype(str))
        history_frame.to_csv(os.path.join(output_dir, 'input_history.csv'), index=False)

        forecast_frame = pd.DataFrame(preds, columns=pred_columns)
        forecast_frame.insert(0, 'date', future_dates.astype(str))
        forecast_frame.to_csv(os.path.join(output_dir, 'forecast.csv'), index=False)

        np.save(os.path.join(output_dir, 'forecast.npy'), preds)

        history_target = history_frame[pred_columns[-1]].to_numpy()
        forecast_target = forecast_frame[pred_columns[-1]].to_numpy()
        combined = np.concatenate((history_target, forecast_target), axis=0)
        visual(combined, None, os.path.join(output_dir, 'forecast.pdf'))

        with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as summary_file:
            summary_file.write(setting + '\n')
            summary_file.write(f'prediction only: no ground truth available\n')
            summary_file.write(f'history_points:{len(history_frame)}, forecast_points:{len(forecast_frame)}\n')
            summary_file.write(f'forecast_target:{pred_columns[-1]}\n')

        return

    def _build_predict_inputs(self):
        file_path = os.path.join(self.args.root_path, self.args.data_path)
        df_raw = pd.read_csv(file_path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        cols = list(df_raw.columns)
        cols.remove('date')
        if self.args.target in cols:
            cols.remove(self.args.target)
        ordered_cols = ['date'] + cols + [self.args.target]
        df_raw = df_raw[ordered_cols]

        if self.args.features in ['M', 'MS']:
            feature_cols = df_raw.columns[1:]
            df_data = df_raw[feature_cols]
        else:
            feature_cols = [self.args.target]
            df_data = df_raw[[self.args.target]]

        if len(df_data) < self.args.seq_len:
            raise ValueError(f'not enough rows for prediction: need at least seq_len={self.args.seq_len}, got {len(df_data)}')

        history_values = df_data.iloc[-self.args.seq_len:].values.astype(np.float32)
        history_dates = df_raw['date'].iloc[-self.args.seq_len:].to_numpy()

        future_dates = pd.date_range(
            start=df_raw['date'].iloc[-1],
            periods=self.args.pred_len + 1,
            freq=self.args.freq,
        )[1:].to_numpy()

        label_len = max(int(self.args.label_len), 0)
        label_dates = pd.to_datetime(history_dates[-label_len:]) if label_len > 0 else pd.to_datetime([])
        decoder_dates = np.concatenate([label_dates.to_numpy(), future_dates], axis=0) if label_len > 0 else future_dates

        if self.args.embed != 'timeF':
            history_stamp = self._build_calendar_features(pd.to_datetime(history_dates))
            decoder_stamp = self._build_calendar_features(pd.to_datetime(decoder_dates))
        else:
            history_stamp = time_features(pd.to_datetime(history_dates), freq=self.args.freq).transpose(1, 0)
            decoder_stamp = time_features(pd.to_datetime(decoder_dates), freq=self.args.freq).transpose(1, 0)

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

        return batch_x, batch_x_mark, dec_inp, batch_y_mark, history_dates, future_dates, list(feature_cols)

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
