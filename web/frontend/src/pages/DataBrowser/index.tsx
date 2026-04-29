// web/frontend/src/pages/DataBrowser/index.tsx
import React, { useEffect, useState } from 'react';
import { Card, Table, Input, Button, Space, Drawer } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import axios from 'axios';

const DataBrowser: React.FC = () => {
  const [data, setData] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [detailVisible, setDetailVisible] = useState(false);
  const [detailCode, setDetailCode] = useState('');
  const [detailData, setDetailData] = useState([]);

  const fetch = async (p: number = 1) => {
    setLoading(true);
    try {
      const res = await axios.get('/api/data/stocks', {
        params: { page: p, search: search || undefined, page_size: 50 },
      });
      setData(res.data.data);
      setTotal(res.data.total);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetch(page); }, [page]);

  const handleDetail = async (code: string) => {
    setDetailCode(code);
    setDetailVisible(true);
    const res = await axios.get(`/api/data/stocks/${code}`);
    setDetailData(res.data);
  };

  return (
    <Card
      title="全量数据浏览"
      extra={
        <Space>
          <Input.Search
            placeholder="代码/名称" prefix={<SearchOutlined />}
            value={search} onChange={(e) => setSearch(e.target.value)}
            onSearch={() => { setPage(1); fetch(1); }}
            style={{ width: 200 }}
          />
          <Button onClick={() => fetch(page)} loading={loading}>刷新</Button>
        </Space>
      }
    >
      <Table
        dataSource={data} rowKey="stock_code" loading={loading}
        pagination={{ total, current: page, pageSize: 50, onChange: setPage }}
        columns={[
          { title: '代码', dataIndex: 'stock_code', width: 100 },
          { title: '名称', dataIndex: 'stock_name', width: 100 },
          { title: '行业', dataIndex: 'industry', width: 100 },
          { title: '日期', dataIndex: 'trade_date', width: 100 },
          { title: '收盘价', dataIndex: 'close', width: 90 },
          {
            title: '涨跌幅', dataIndex: 'change_pct', width: 80,
            render: (v: number) => <span style={{ color: v >= 0 ? '#cf1322' : '#3f8600' }}>{v}%</span>,
          },
          { title: '换手率', dataIndex: 'turnover_rate', width: 80 },
          { title: 'PE', dataIndex: 'pe_ttm', width: 70 },
          { title: 'PB', dataIndex: 'pb', width: 70 },
          { title: '流通市值(亿)', dataIndex: 'circulating_mv', width: 110, render: (v: number) => (v / 1e8).toFixed(1) },
          {
            title: '操作', width: 80, fixed: 'right' as const,
            render: (_: any, r: any) => <Button size="small" onClick={() => handleDetail(r.stock_code)}>详情</Button>,
          },
        ]}
        scroll={{ x: 1000 }}
      />

      <Drawer
        title={`${detailCode} 历史数据`} open={detailVisible}
        onClose={() => setDetailVisible(false)} width={800}
      >
        <Table
          dataSource={detailData} rowKey={(r: any) => `${r.stock_code}-${r.trade_date}`}
          pagination={{ pageSize: 20 }}
          columns={[
            { title: '日期', dataIndex: 'trade_date', width: 100 },
            { title: '开', dataIndex: 'open', width: 70 }, { title: '高', dataIndex: 'high', width: 70 },
            { title: '低', dataIndex: 'low', width: 70 }, { title: '收', dataIndex: 'close', width: 70 },
            { title: '量', dataIndex: 'volume', width: 90 },
            { title: 'PE', dataIndex: 'pe_ttm', width: 60 }, { title: 'PB', dataIndex: 'pb', width: 60 },
          ]}
          scroll={{ x: 600 }}
        />
      </Drawer>
    </Card>
  );
};

export default DataBrowser;