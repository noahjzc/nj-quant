// web/frontend/src/pages/Positions/index.tsx
import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Table, Tag, Button, Modal, InputNumber, Space, message } from 'antd';
import { WalletOutlined, DollarOutlined, StockOutlined, RiseOutlined } from '@ant-design/icons';
import axios from 'axios';

const Positions: React.FC = () => {
  const [overview, setOverview] = useState<any>({});
  const [positions, setPositions] = useState([]);
  const [history, setHistory] = useState([]);
  const [capitalLogs, setCapitalLogs] = useState([]);
  const [depositVisible, setDepositVisible] = useState(false);
  const [depositAmount, setDepositAmount] = useState<number>();
  const [loading, setLoading] = useState(false);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [ov, pos, hist, cap] = await Promise.all([
        axios.get('/api/positions/overview'),
        axios.get('/api/positions/', { params: { status: 'OPEN' } }),
        axios.get('/api/positions/', { params: { status: 'CLOSED' } }),
        axios.get('/api/positions/capital'),
      ]);
      setOverview(ov.data);
      setPositions(pos.data);
      setHistory(hist.data);
      setCapitalLogs(cap.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAll(); }, []);

  const handleDeposit = async () => {
    if (!depositAmount) return;
    await axios.post('/api/positions/capital/deposit', { amount: depositAmount });
    message.success('资金补充成功');
    setDepositVisible(false);
    setDepositAmount(undefined);
    fetchAll();
  };

  const posColumns = [
    { title: '代码', dataIndex: 'stock_code', width: 100 },
    { title: '名称', dataIndex: 'stock_name', width: 100 },
    { title: '买入日', dataIndex: 'buy_date', width: 100 },
    {
      title: '买入价', dataIndex: 'buy_price', width: 90,
      render: (v: number) => v?.toFixed(2),
    },
    { title: '股数', dataIndex: 'shares', width: 80 },
    {
      title: '市值', key: 'mv', width: 100,
      render: (_: any, r: any) => (r.shares * r.buy_price).toFixed(2),
    },
  ];

  const historyColumns = [
    ...posColumns.filter(c => c.dataIndex !== 'shares'),
    {
      title: '卖出日', dataIndex: 'sell_date', width: 100,
    },
    {
      title: '卖出价', dataIndex: 'sell_price', width: 90,
      render: (v: number) => v?.toFixed(2),
    },
    {
      title: '收益率', dataIndex: 'profit_pct', width: 90,
      render: (v: number) => (
        <Tag color={v >= 0 ? 'red' : 'green'}>
          {v != null ? `${v.toFixed(2)}%` : '-'}
        </Tag>
      ),
    },
  ];

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card><Statistic title="总资产" value={overview.total_asset} precision={2} prefix={<WalletOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="可用资金" value={overview.available_cash} precision={2} prefix={<DollarOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="持仓市值" value={overview.position_value} precision={2} prefix={<StockOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="累计收益" value={overview.total_profit} precision={2} prefix={<RiseOutlined />} /></Card>
        </Col>
      </Row>

      <Card
        title="当前持仓"
        extra={
          <Space>
            <Button type="primary" onClick={() => setDepositVisible(true)}>补充资金</Button>
            <Button onClick={fetchAll} loading={loading}>刷新</Button>
          </Space>
        }
        style={{ marginBottom: 24 }}
      >
        <Table columns={posColumns} dataSource={positions} rowKey="id" pagination={false}
          locale={{ emptyText: '暂无持仓' }} />
      </Card>

      <Card title="历史交易" style={{ marginBottom: 24 }}>
        <Table columns={historyColumns} dataSource={history} rowKey="id"
          pagination={{ pageSize: 10 }} />
      </Card>

      <Card title="资金流水">
        <Table
          dataSource={capitalLogs} rowKey="id" pagination={{ pageSize: 10 }}
          columns={[
            { title: '类型', dataIndex: 'event_type', width: 80 },
            {
              title: '金额', dataIndex: 'amount', width: 120,
              render: (v: number) => <span style={{ color: v >= 0 ? '#3f8600' : '#cf1322' }}>{v.toFixed(2)}</span>,
            },
            { title: '余额', dataIndex: 'balance_after', width: 120, render: (v: number) => v?.toFixed(2) },
            { title: '备注', dataIndex: 'note' },
            { title: '时间', dataIndex: 'created_at', width: 170 },
          ]}
        />
      </Card>

      <Modal
        title="补充资金"
        open={depositVisible}
        onOk={handleDeposit}
        onCancel={() => setDepositVisible(false)}
      >
        <InputNumber
          style={{ width: '100%' }} min={0} step={1000}
          value={depositAmount} onChange={(v) => setDepositAmount(v || undefined)}
          placeholder="金额"
        />
      </Modal>
    </div>
  );
};

export default Positions;