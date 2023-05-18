# -*- coding: UTF-8 -*-

""" TimelyRec
Reference:
    "Learning Heterogeneous Temporal Patterns of User Preference for Timely Recommendation"
    Junsu Cho et al., WWW'2021.
CMD example:
    python main.py --model_name TimelyRec --lr 0.001 --batch_size 256 --dropout 0.1 --dataset 'ml_1m_timely_rec'
"""

import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel


class TimelyRec(SequentialModel):
    extra_log_args = ['emb_size', 'seq_len', 'width', 'depth']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32)
        parser.add_argument('--seq_len', type=int, default=5)
        parser.add_argument('--width', type=int, default=128)
        parser.add_argument('--depth', type=int, default=4)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.seq_len = args.seq_len
        self.width = args.width
        self.depth = args.depth
        super().__init__(args, corpus)

    def _define_params(self):

        self.userEmbedding = nn.Embedding(self.user_num + 1, self.emb_size)
        self.itemEmbeddingSet = nn.Embedding(self.item_num + 1, self.emb_size)

        self.monthEmbedding = nn.Embedding(12, self.emb_size)
        self.dayEmbedding = nn.Embedding(7, self.emb_size)
        self.dateEmbedding = nn.Embedding(31, self.emb_size)
        self.hourEmbedding = nn.Embedding(24, self.emb_size)

        self.meanLayer = meanLayer()
        self.PositionalEncoding = PositionalEncoding(self.emb_size)
        self.TemporalPositionEncoding = TemporalPositionEncoding()
        self.MATE = MATE(self.emb_size)
        self.TAHE = TAHE(self.emb_size)

    def forward(self, feed_dict):
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        userInput = feed_dict['user_id']  # [batch_size, 1]
        itemInput = feed_dict['item_id'][:, 0]
        monthInput = feed_dict['month']
        dayInput = feed_dict['day_of_week']
        dateInput = feed_dict['date']
        hourInput = feed_dict['hour']
        curTimestampInput = feed_dict['time']
        batch_size = itemInput.shape[0]

        recentMonthInput = [feed_dict[f'month{i}'] for i in range(self.seq_len)]
        recentDayInput = [feed_dict[f'day_of_week{i}'] for i in range(self.seq_len)]
        recentDateInput = [feed_dict[f'date{i}'] for i in range(self.seq_len)]
        recentHourInput = [feed_dict[f'hour{i}'] for i in range(self.seq_len)]
        recentTimestampInput = [feed_dict[f'time{i}'] for i in range(self.seq_len)]
        recentItemidInput = [feed_dict[f'item_id{i}'] for i in range(self.seq_len)]

        userEmbedding = self.userEmbedding(userInput.reshape(-1, 1))
        itemEmbedding = self.itemEmbeddingSet(itemInput.reshape(-1, 1))
        recentItemEmbeddings = self.itemEmbeddingSet(torch.stack(recentItemidInput, dim=1))
        recentTimestamps = torch.stack(recentTimestampInput, dim=1)

        curMonthEmbedding = self.monthEmbedding(monthInput.reshape(-1, 1))
        curDayEmbedding = self.dayEmbedding(dayInput.reshape(-1, 1))
        curDateEmbedding = self.dateEmbedding(dateInput.reshape(-1, 1))
        curHourEmbedding = self.hourEmbedding(hourInput.reshape(-1, 1))

        recentMonthEmbeddings = self.monthEmbedding(torch.stack(recentMonthInput, dim=1))
        recentDayEmbeddings = self.dayEmbedding(torch.stack(recentDayInput, dim=1))
        recentDateEmbeddings = self.dateEmbedding(torch.stack(recentDateInput, dim=1))
        recentHourEmbeddings = self.hourEmbedding(torch.stack(recentHourInput, dim=1))

        prevMonthEmbeddings = []
        prevDayEmbeddings = []
        prevDateEmbeddings = []
        prevHourEmbeddings = []

        ratio = 0.2
        for i in range(self.seq_len):
            prevMonthEmbeddings.append([])
            for j in range(1, max(int(12 * ratio + 0.5), 1) + 1):
                monthSurr = self.monthEmbedding(SurroundingSlots(window_length=j, max_range=12)(recentMonthInput[i]))
                prevMonthEmbeddings[i].append(self.meanLayer(monthSurr))

            prevDayEmbeddings.append([])
            for j in range(1, max(int(7 * ratio + 0.5), 1) + 1):
                daySurr = self.dayEmbedding(SurroundingSlots(window_length=j, max_range=7)(recentDayInput[i]))
                prevDayEmbeddings[i].append(self.meanLayer(daySurr))

            prevDateEmbeddings.append([])
            for j in range(1, max(int(31 * ratio + 0.5), 1) + 1):
                dateSurr = self.dateEmbedding(SurroundingSlots(window_length=j, max_range=31)(recentDateInput[i]))
                prevDateEmbeddings[i].append(self.meanLayer(dateSurr))

            prevHourEmbeddings.append([])
            for j in range(1, max(int(24 * ratio + 0.5), 1) + 1):
                hourSurr = self.hourEmbedding(SurroundingSlots(window_length=j, max_range=24)(recentHourInput[i]))
                prevHourEmbeddings[i].append(self.meanLayer(hourSurr))
        
        monthEmbeddings = []
        dayEmbeddings = []
        dateEmbeddings = []
        hourEmbeddings = []

        for i in range(1, max(int(12 * ratio + 0.5), 1) + 1):
            monthSurr = self.monthEmbedding(SurroundingSlots(window_length=i, max_range=12)(monthInput))
            monthEmbeddings.append(self.meanLayer(monthSurr))
            
        for i in range(1, max(int(7 * ratio + 0.5), 1) + 1):
            daySurr = self.dayEmbedding(SurroundingSlots(window_length=i, max_range=7)(dayInput))
            dayEmbeddings.append(self.meanLayer(daySurr))
            
        for i in range(1, max(int(31 * ratio + 0.5), 1) + 1):
            dateSurr = self.dateEmbedding(SurroundingSlots(window_length=i, max_range=31)(dateInput))
            dateEmbeddings.append(self.meanLayer(dateSurr))
            
        for i in range(1, max(int(24 * ratio + 0.5), 1) + 1):
            hourSurr = self.hourEmbedding(SurroundingSlots(window_length=i, max_range=24)(hourInput))
            hourEmbeddings.append(self.meanLayer(hourSurr))


        if int(12 * ratio + 0.5) <= 1:
            monthEmbeddings = monthEmbeddings[0]
            for i in range(self.seq_len):
                prevMonthEmbeddings[i] = prevMonthEmbeddings[i][0]
        else:
            monthEmbeddings = torch.cat(monthEmbeddings, dim=1)
            for i in range(self.seq_len):
                prevMonthEmbeddings[i] = torch.cat(prevMonthEmbeddings[i], dim=1)

        if int(7 * ratio + 0.5) <= 1:
            dayEmbeddings = dayEmbeddings[0]
            for i in range(self.seq_len):
                prevDayEmbeddings[i] = prevDayEmbeddings[i][0]
        else:
            dayEmbeddings = torch.cat(dayEmbeddings, dim=1)
            for i in range(self.seq_len):
                prevDayEmbeddings[i] = torch.cat(prevDayEmbeddings[i], dim=1)

        if int(31 * ratio + 0.5) <= 1:
            dateEmbeddings = dateEmbeddings[0]
            for i in range(self.seq_len):
                prevDateEmbeddings[i] = prevDateEmbeddings[i][0]
        else:
            dateEmbeddings = torch.cat(dateEmbeddings, dim=1)
            for i in range(self.seq_len):
                prevDateEmbeddings[i] = torch.cat(prevDateEmbeddings[i], dim=1)

        if int(24 * ratio + 0.5) <= 1:
            hourEmbeddings = hourEmbeddings[0]
            for i in range(self.seq_len):
                prevHourEmbeddings[i] = prevHourEmbeddings[i][0]
        else:
            hourEmbeddings = torch.cat(hourEmbeddings, dim=1)
            for i in range(self.seq_len):
                prevHourEmbeddings[i] = torch.cat(prevHourEmbeddings[i], dim=1)
        
        recentTimestampTEs = self.PositionalEncoding(recentTimestamps)
        curTimestampTE = self.PositionalEncoding(curTimestampInput.reshape(-1, 1))

        # temporal position encoding
        itemEmbedding = self.TemporalPositionEncoding([itemEmbedding, curTimestampTE])    
        recentItemEmbeddings = self.TemporalPositionEncoding([recentItemEmbeddings, recentTimestampTEs])

        userVector = torch.squeeze(userEmbedding)
        itemVector = torch.squeeze(itemEmbedding)
        curTimestampTE = torch.squeeze(curTimestampTE)
        
        # MATE
        curTimeRepresentation = torch.squeeze(self.MATE([userEmbedding, curMonthEmbedding, curDayEmbedding, curDateEmbedding, curHourEmbedding, monthEmbeddings, dayEmbeddings, dateEmbeddings, hourEmbeddings])) # None * embedding_size
        prevTimeRepresentations = []
        for i in range(self.seq_len):
            prevTimeRepresentations.append(self.MATE([userEmbedding, Slice(i)(recentMonthEmbeddings), Slice(i)(recentDayEmbeddings), Slice(i)(recentDateEmbeddings), Slice(i)(recentHourEmbeddings), prevMonthEmbeddings[i], prevDayEmbeddings[i], prevDateEmbeddings[i], prevHourEmbeddings[i]])) # None * embedding_size)
        prevTimeRepresentations = torch.cat(prevTimeRepresentations, dim=1)

        # TAHE
        userHistoryRepresentation = self.TAHE([prevTimeRepresentations, curTimeRepresentation, recentTimestamps, recentItemEmbeddings])

        # combination
        x = torch.cat([userVector, itemVector, curTimeRepresentation, userHistoryRepresentation], dim=-1).float()
        in_shape = self.emb_size * 4

        for i in range(self.depth):
            if i == self.depth - 1:
                x = nn.Linear(in_shape, 1)(x)
            else:
                x = nn.Linear(in_shape, self.width)(x)
                x = x.relu()
                if self.dropout is not None:
                    x = nn.Dropout(self.dropout)(x)
            in_shape = self.width
        
        # x = x.sigmoid()
        pred_vectors = self.itemEmbeddingSet(i_ids)
        prediction = (x[:, None, :] * pred_vectors).sum(-1)

        return {'prediction': prediction.view(batch_size, -1)}


class SurroundingSlots(nn.Module):
    def __init__(self, window_length, max_range):
        super().__init__()
        self.window_length = window_length
        self.max_range = max_range

    def forward(self, x):
        surr = x.long()[:, None] + torch.arange(-self.window_length, self.window_length + 1)
        surrUnderflow = (surr < 0).long()
        surrOverflow = (surr > self.max_range - 1).long()
        return surr * (-(surrUnderflow + surrOverflow) + 1) + surrUnderflow * (surr + self.max_range) + surrOverflow * (surr - self.max_range)


class MATE(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

        # for multiplicative attention
        self.W = nn.Parameter(nn.init.normal_(torch.empty(self.dimension, self.dimension)))

        # for personalization
        self.Wmonth = nn.Parameter(nn.init.normal_(torch.empty(self.dimension, self.dimension)))
        self.Wday = nn.Parameter(nn.init.normal_(torch.empty(self.dimension, self.dimension)))
        self.Wdate = nn.Parameter(nn.init.normal_(torch.empty(self.dimension, self.dimension)))
        self.Whour = nn.Parameter(nn.init.normal_(torch.empty(self.dimension, self.dimension)))

    def forward(self, x):
        userEmbedding = x[0]

        curMonthEmbedding = x[1].view(-1, 1, self.dimension)
        curDayEmbedding = x[2].view(-1, 1, self.dimension)
        curDateEmbedding = x[3].view(-1, 1, self.dimension)
        curHourEmbedding = x[4].view(-1, 1, self.dimension)

        monthEmbeddings = x[5]
        dayEmbeddings = x[6]
        dateEmbeddings = x[7]
        hourEmbeddings = x[8]

        # personalization
        curMonthEmbedding = curMonthEmbedding * (torch.matmul(userEmbedding, self.Wmonth))
        curDayEmbedding = curDayEmbedding * (torch.matmul(userEmbedding, self.Wday))
        curDateEmbedding = curDateEmbedding * (torch.matmul(userEmbedding, self.Wdate))
        curHourEmbedding = curHourEmbedding * (torch.matmul(userEmbedding, self.Whour))
        monthEmbeddings = monthEmbeddings * (torch.matmul(userEmbedding, self.Wmonth))
        dayEmbeddings = dayEmbeddings * (torch.matmul(userEmbedding, self.Wday))
        dateEmbeddings = dateEmbeddings * (torch.matmul(userEmbedding, self.Wdate))
        hourEmbeddings = hourEmbeddings * (torch.matmul(userEmbedding, self.Whour))

        # query for gradated attention
        monthQ = curMonthEmbedding
        dayQ = curDayEmbedding
        dateQ = curDateEmbedding
        hourQ = curHourEmbedding
        
        # key, value
        monthKV = torch.cat([monthEmbeddings, curMonthEmbedding], dim=1)
        dayKV = torch.cat([dayEmbeddings, curDayEmbedding], dim=1)
        dateKV = torch.cat([dateEmbeddings, curDateEmbedding], dim=1)
        hourKV = torch.cat([hourEmbeddings, curHourEmbedding], dim=1)

        # attention score
        monthQKV = (torch.matmul(monthQ, monthKV.permute(0, 2, 1)) / np.sqrt(self.dimension)).softmax(dim=-1)
        dayQKV = (torch.matmul(dayQ, dayKV.permute(0, 2, 1)) / np.sqrt(self.dimension)).softmax(dim=-1)
        dateQKV = (torch.matmul(dateQ, dateKV.permute(0, 2, 1)) / np.sqrt(self.dimension)).softmax(dim=-1)
        hourQKV = (torch.matmul(hourQ, hourKV.permute(0, 2, 1)) / np.sqrt(self.dimension)).softmax(dim=-1)

        # embedding for each granularity of period information
        monthEmbedding = torch.matmul(monthQKV, monthKV)
        dayEmbedding = torch.matmul(dayQKV, dayKV)
        dateEmbedding = torch.matmul(dateQKV, dateKV)
        hourEmbedding = torch.matmul(hourQKV, hourKV)

        # multiplicative attention
        q = userEmbedding
        kv = torch.cat([monthEmbedding, dayEmbedding, dateEmbedding, hourEmbedding], dim=1)
        qW = torch.matmul(q, self.W)
        a = torch.sigmoid(torch.matmul(qW, kv.permute(0, 2, 1)))
        timeRepresentation = torch.matmul(a, kv)

        return timeRepresentation


class TAHE(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        
    def forward(self, x):
        recentTimeRepresentations = x[0]
        curTimeRepresentation = x[1]
        recentTimestamps = x[2]
        recentItemEmbeddings = x[3] 

        # previous timestamp == 0 ==> no history
        mask = (recentTimestamps > 0).double()[:, None, :]

        # time-based attention
        similarity = torch.bmm(nn.functional.normalize(curTimeRepresentation, dim=-1)[:, None, :], nn.functional.normalize(recentTimeRepresentations, dim=-1).permute(0, 2, 1))
        masked_similarity = mask * ((similarity + 1.0) / 2.0)
        weightedPrevItemEmbeddings = torch.bmm(masked_similarity, recentItemEmbeddings)
        userHistoryRepresentation = torch.squeeze(weightedPrevItemEmbeddings)

        return userHistoryRepresentation


class meanLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1, keepdims=True)


class Slice(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    
    def forward(self, x):
        return x[:, self.index, :]


class TemporalPositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(nn.init.ones_(torch.empty(1)))

    def forward(self, x):
        item = x[0]
        time = x[1]
        return item + time * self.a


class PositionalEncoding(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        x = (x / 3600).long().double()

        evens = torch.reshape(torch.stack([torch.tensor([0.0, 1.0])] * int(self.output_dim / 2), dim=0), shape=(-1,))
        odds = torch.reshape(torch.stack([torch.tensor([1.0, 0.0])] * int(self.output_dim / 2), dim=0), shape=(-1,))
        pos = torch.reshape(torch.pow(10000.0, ((torch.arange(self.output_dim) / 2) * 2).double() / self.output_dim), shape=(1, -1))
        pos = torch.reshape(pos.repeat(x.size(0) * x.size(1), 1), shape=(x.size(0), x.size(1), -1))
        evenEmb = torch.sin(x[:, :, None].repeat(1, 1, self.output_dim) / pos) * evens
        oddEmb = torch.cos(x[:, :, None].repeat(1, 1, self.output_dim) / pos) * odds
        posEmbedding = evenEmb + oddEmb

        return posEmbedding
