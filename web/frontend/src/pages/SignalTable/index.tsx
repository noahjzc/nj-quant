import React, { useState, useEffect } from 'react';
import { Table, Tag, Button, Space, message, Modal, Tooltip, Card, Row, Col, InputNumber } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, QuestionCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import api from '../../utils/api';
import dayjs from 'dayjs';

interface SignalRecord {
  id: number;
  trade_date: string;
  stock_code: string;
  stock_name: string | null;
  direction: string;
  target_pct: number | null;
  price_low: number | null;
  price_high: number | null;
  signal_reason: string | null;
  status: string;
  executed_price: number | null;
  confirmed_at: string | null;
  created_at: string | null;
}

const SignalTable: React.FC = () => {
  const [data, setData] = useState<SignalRecord[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      const response = await api.get('/signals');
      setData(response.data);
    } catch {
      message.error('获取信号数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const handleConfirm = async (record: SignalRecord) => {
    let price: number | null = null;

    Modal.confirm({
      title: '确认信号执行',
      content: (
        <div>
          <p>股票: {record.stock_name} ({record.stock_code})</p>
          <p>方向: {record.direction}</p>
          <p>目标仓位: {record.target_pct}%</p>
          <p>参考价: {record.price_low} ~ {record.price_high}</p>
          <InputNumber
            style={{ width: '100%', marginTop: 8 }}
            placeholder="输入实际成交价"
            min={0}
            precision={2}
            onChange={(v) => { price = v ?? null; }}
          />
        </div>
      ),
      onOk: async () => {
        if (price === null) {
          message.warning('请输入实际成交价');
          return false;
        }
        try {
          await api.post(`/signals/${record.id}/confirm`, { executed_price: price });
          message.success('已确认');
          fetchData();
        } catch {
          message.error('确认失败');
        }
      },
    });
  };

  const handleSkip = async (record: SignalRecord) => {
    try {
      await api.post(`/signals/${record.id}/skip`);
      message.success('已跳过');
      fetchData();
    } catch {
      message.error('操作失败');
    }
  };

  const columns = [
    {
      title: '交易日期',
      dataIndex: 'trade_date',
      key: 'trade_date',
      width: 110,
    },
    {
      title: '股票代码',
      dataIndex: 'stock_code',
      key: 'stock_code',
      width: 100,
      fixed: 'left' as const,
    },
    {
      title: '股票名称',
      dataIndex: 'stock_name',
      key: 'stock_name',
      width: 100,
    },
    {
      title: '方向',
      dataIndex: 'direction',
      key: 'direction',
      width: 80,
      render: (d: string) => (
        <Tag color={d === 'BUY' ? 'green' : 'red'}>{d === 'BUY' ? '买入' : '卖出'}</Tag>
      ),
    },
    {
      title: '信号原因',
      dataIndex: 'signal_reason',
      key: 'signal_reason',
      width: 120,
      ellipsis: true,
      render: (v: string | null) => v || '-',
    },
    {
      title: '目标仓位',
      dataIndex: 'target_pct',
      key: 'target_pct',
      width: 90,
      render: (v: number | null) => (v != null ? `${v}%` : '-'),
    },
    {
      title: '参考价格区间',
      key: 'price_range',
      width: 130,
      render: (_: unknown, r: SignalRecord) =>
        r.price_low != null && r.price_high != null
          ? `${r.price_low} ~ ${r.price_high}`
          : '-',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 90,
      render: (s: string) => {
        const colorMap: Record<string, string> = { pending: 'gold', confirmed: 'green', skipped: 'gray' };
        const textMap: Record<string, string> = { pending: '待确认', confirmed: '已确认', skipped: '已跳过' };
        return <Tag color={colorMap[s]}>{textMap[s] || s}</Tag>;
      },
    },
    {
      title: '确认时间',
      dataIndex: 'confirmed_at',
      key: 'confirmed_at',
      width: 170,
      render: (t: string | null) => (t ? dayjs(t).format('YYYY-MM-DD HH:mm:ss') : '-'),
    },
    {
      title: '操作',
      key: 'action',
      width: 150,
      fixed: 'right' as const,
      render: (_: unknown, record: SignalRecord) => {
        if (record.status !== 'pending') {
          return <span style={{ color: '#999' }}>已处理</span>;
        }
        return (
          <Space size="small">
            <Tooltip title="确认执行">
              <Button type="link" icon={<CheckCircleOutlined />} onClick={() => handleConfirm(record)} style={{ color: '#52c41a', padding: '0 4px' }} />
            </Tooltip>
            <Tooltip title="跳过">
              <Button type="link" icon={<CloseCircleOutlined />} onClick={() => handleSkip(record)} style={{ color: '#ff4d4f', padding: '0 4px' }} />
            </Tooltip>
            <Tooltip title="详情">
              <Button
                type="link"
                icon={<QuestionCircleOutlined />}
                onClick={() => Modal.info({
                  title: '信号详情',
                  content: (
                    <div>
                      <p>股票: {record.stock_name} ({record.stock_code})</p>
                      <p>方向: {record.direction}</p>
                      <p>交易日期: {record.trade_date}</p>
                      <p>信号原因: {record.signal_reason || '-'}</p>
                      <p>目标仓位: {record.target_pct != null ? `${record.target_pct}%` : '-'}</p>
                      <p>参考价格: {record.price_low != null && record.price_high != null ? `${record.price_low} ~ ${record.price_high}` : '-'}</p>
                    </div>
                  ),
                })}
                style={{ padding: '0 4px' }}
              />
            </Tooltip>
          </Space>
        );
      },
    },
  ];

  const pending = data.filter((d) => d.status === 'pending').length;
  const confirmed = data.filter((d) => d.status === 'confirmed').length;
  const skipped = data.filter((d) => d.status === 'skipped').length;

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small" title="待处理信号"><h2 style={{ margin: 0 }}>{pending}</h2></Card>
        </Col>
        <Col span={6}>
          <Card size="small" title="已确认"><h2 style={{ margin: 0 }}>{confirmed}</h2></Card>
        </Col>
        <Col span={6}>
          <Card size="small" title="已跳过"><h2 style={{ margin: 0 }}>{skipped}</h2></Card>
        </Col>
        <Col span={6}>
          <Card size="small" title="总计"><h2 style={{ margin: 0 }}>{data.length}</h2></Card>
        </Col>
      </Row>

      <Space style={{ marginBottom: 16 }}>
        <Button icon={<ReloadOutlined />} onClick={fetchData} loading={loading}>刷新</Button>
      </Space>

      <Table
        columns={columns}
        dataSource={data}
        loading={loading}
        rowKey="id"
        scroll={{ x: 1300 }}
        pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
      />
    </div>
  );
};

export default SignalTable;
