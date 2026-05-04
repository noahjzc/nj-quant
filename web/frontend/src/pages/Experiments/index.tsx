import React, { useState, useEffect, useCallback } from 'react';
import { Table, Select, Button, Modal, Tag, Space, message } from 'antd';
import api from '../../utils/api';

interface Experiment {
  experiment_id: string;
  timestamp: string;
  type: string;
  ranker: string;
  date_range: { start: string; end: string };
  metrics: Record<string, number>;
  factor_count: number;
}

interface CompareRow {
  metric?: string;
  param?: string;
  values: (number | null)[];
  has_diff: boolean;
}

const METRIC_LABELS: Record<string, string> = {
  sharpe: 'Sharpe', annual_return: '年化收益', max_drawdown: '最大回撤',
  calmar: 'Calmar', win_rate: '胜率', ic: 'IC', total_return: '总收益',
};

const TYPE_MAP: Record<string, string> = {
  backtest: '回测', optimization_single: '单期优化',
  optimization_wf: 'Walk-Forward', sensitivity: '敏感性分析',
};

const RANKER_MAP: Record<string, string> = {
  SignalRanker: 'Signal', MLRanker: 'LightGBM', TemporalMLRanker: 'Temporal+LGB',
};

const ExperimentsPage: React.FC = () => {
  const [data, setData] = useState<Experiment[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareData, setCompareData] = useState<{
    experiments: any[]; metrics_table: CompareRow[]; params_table: CompareRow[];
  } | null>(null);
  const [filters, setFilters] = useState({ ranker: '', type: '', sort: 'timestamp', order: 'desc' });
  const [page, setPage] = useState(1);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const params: any = { limit: 20, offset: (page - 1) * 20, sort: filters.sort, order: filters.order };
      if (filters.ranker) params.ranker = filters.ranker;
      if (filters.type) params.type = filters.type;
      const res = await api.get('/experiments/', { params });
      setData(res.data.items);
      setTotal(res.data.total);
    } catch (e) { message.error('加载实验列表失败'); }
    setLoading(false);
  }, [page, filters]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleCompare = async () => {
    if (selected.length < 2) return;
    const res = await api.post('/experiments/compare', { ids: selected });
    setCompareData(res.data);
    setCompareOpen(true);
  };

  const formatPct = (v: number | null | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '-';
  const formatNum = (v: number | null | undefined) => v != null ? v.toFixed(4) : '-';

  const columns: any[] = [
    { title: 'ID', dataIndex: 'experiment_id', width: 140 },
    { title: '时间', dataIndex: 'timestamp', width: 160, render: (v: string) => v?.replace('T', ' ').substring(0, 16) },
    { title: '类型', dataIndex: 'type', width: 90, render: (v: string) => TYPE_MAP[v] || v },
    {
      title: '模型', dataIndex: 'ranker', width: 150,
      render: (v: string) => <Tag color={v === 'TemporalMLRanker' ? 'blue' : v === 'MLRanker' ? 'green' : 'default'}>{RANKER_MAP[v] || v}</Tag>,
    },
    { title: 'Sharpe', dataIndex: ['metrics', 'sharpe'], width: 80, sorter: true, render: formatNum },
    { title: '年化收益', dataIndex: ['metrics', 'annual_return'], width: 90, sorter: true, render: formatPct },
    { title: '最大回撤', dataIndex: ['metrics', 'max_drawdown'], width: 90, sorter: true, render: formatPct },
    { title: 'Calmar', dataIndex: ['metrics', 'calmar'], width: 80, sorter: true, render: formatNum },
    { title: '胜率', dataIndex: ['metrics', 'win_rate'], width: 70, sorter: true, render: formatPct },
    { title: 'IC', dataIndex: ['metrics', 'ic'], width: 70, sorter: true, render: formatNum },
    { title: '因字数', dataIndex: 'factor_count', width: 60, sorter: true },
  ];

  const isPctMetric = (key: string) => ['annual_return', 'max_drawdown', 'win_rate', 'total_return'].includes(key);
  const isBestMetric = (key: string) => key !== 'max_drawdown';

  return (
    <div>
      <h2>实验追踪</h2>
      <Space style={{ marginBottom: 16 }}>
        <Select placeholder="模型类型" allowClear style={{ width: 180 }}
          value={filters.ranker || undefined}
          onChange={(v) => { setFilters({ ...filters, ranker: v || '' }); setPage(1); }}>
          <Select.Option value="">全部</Select.Option>
          <Select.Option value="TemporalMLRanker">Temporal+LightGBM</Select.Option>
          <Select.Option value="MLRanker">LightGBM</Select.Option>
          <Select.Option value="SignalRanker">Signal</Select.Option>
        </Select>
        <Select placeholder="实验类型" allowClear style={{ width: 130 }}
          value={filters.type || undefined}
          onChange={(v) => { setFilters({ ...filters, type: v || '' }); setPage(1); }}>
          <Select.Option value="">全部</Select.Option>
          <Select.Option value="backtest">回测</Select.Option>
          <Select.Option value="optimization_single">单期优化</Select.Option>
          <Select.Option value="optimization_wf">Walk-Forward</Select.Option>
          <Select.Option value="sensitivity">敏感性分析</Select.Option>
        </Select>
        <Button type="primary" disabled={selected.length < 2} onClick={handleCompare}>
          对比选中 ({selected.length})
        </Button>
      </Space>
      <Table
        rowKey="experiment_id"
        rowSelection={{ selectedRowKeys: selected, onChange: (keys) => setSelected(keys as string[]) }}
        columns={columns}
        dataSource={data}
        loading={loading}
        pagination={{ total, pageSize: 20, current: page, onChange: (p) => setPage(p), showTotal: (t) => `共 ${t} 条` }}
        onChange={(_p: any, _f: any, sorter: any) => {
          if (sorter.field) {
            const metricFields: Record<string, string> = {
              sharpe: 'metrics.sharpe', max_drawdown: 'metrics.max_drawdown',
              annual_return: 'metrics.annual_return', calmar: 'metrics.calmar',
              win_rate: 'metrics.win_rate', ic: 'metrics.ic',
            };
            const sortKey = metricFields[sorter.field] || 'factor_count';
            setFilters({ ...filters, sort: sortKey, order: sorter.order === 'ascend' ? 'asc' : 'desc' });
          }
        }}
        size="middle"
      />
      <Modal title="实验对比" open={compareOpen} onCancel={() => setCompareOpen(false)} width={900} footer={null}>
        {compareData && (
          <>
            <div style={{ marginBottom: 16 }}>
              {compareData.experiments.map((e: any, i: number) => (
                <Tag key={i} color="blue" style={{ marginBottom: 4 }}>
                  {e.experiment_id} — {RANKER_MAP[e.ranker] || e.ranker}
                </Tag>
              ))}
            </div>
            <h4>指标对比</h4>
            <Table
              dataSource={compareData.metrics_table}
              columns={[
                { title: '指标', dataIndex: 'metric', width: 120,
                  render: (v: string) => METRIC_LABELS[v] || v },
                ...compareData.experiments.map((e: any, i: number) => ({
                  title: e.experiment_id.substring(4, 16),
                  dataIndex: 'values', width: 100,
                  render: (_: any, row: CompareRow) => {
                    const val = row.values[i];
                    const numVals = row.values.filter((v: any): v is number => typeof v === 'number');
                    const best = isBestMetric(row.metric || '') && typeof val === 'number'
                      ? val === Math.max(...numVals) : typeof val === 'number'
                      ? val === Math.min(...numVals) : false;
                    const style: React.CSSProperties = {
                      color: best ? '#389e0d' : undefined, fontWeight: best ? 'bold' : undefined,
                    };
                    return <span style={style}>{typeof val === 'number' ? (isPctMetric(row.metric || '') ? formatPct(val) : formatNum(val)) : '-'}</span>;
                  },
                  onCell: (record: CompareRow) => ({
                    style: { backgroundColor: record.has_diff ? '#fff7e6' : undefined },
                  }),
                })),
              ]}
              rowKey="metric" size="small" pagination={false} style={{ marginBottom: 24 }}
            />
            <h4>参数对比</h4>
            <Table
              dataSource={compareData.params_table}
              columns={[
                { title: '参数', dataIndex: 'param', width: 180 },
                ...compareData.experiments.map((e: any, i: number) => ({
                  title: e.experiment_id.substring(4, 16),
                  dataIndex: 'values', width: 120,
                  render: (_: any, row: CompareRow) => String(row.values[i] ?? '-'),
                  onCell: (record: CompareRow) => ({
                    style: { backgroundColor: record.has_diff ? '#fff7e6' : undefined },
                  }),
                })),
              ]}
              rowKey="param" size="small" pagination={false}
            />
          </>
        )}
      </Modal>
    </div>
  );
};

export default ExperimentsPage;
