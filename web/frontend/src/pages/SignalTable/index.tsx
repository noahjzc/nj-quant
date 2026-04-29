import React, { useState, useEffect } from 'react';
import { Table, Tag, Button, Space, message, Modal, Tooltip, Card, Row, Col } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, QuestionCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import axios from 'axios';
import dayjs from 'dayjs';

interface SignalRecord {
  id: number;
  stock_code: string;
  stock_name: string;
  signal_type: string;
  signal_time: string;
  price: number;
  confidence: number;
  status: 'pending' | 'confirmed' | 'skipped';
  confirmed_at?: string;
  confirmed_by?: string;
  notes?: string;
}

const SignalTable: React.FC = () => {
  const [data, setData] = useState<SignalRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [pagination, setPagination] = useState({ current: 1, pageSize: 20, total: 0 });
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/signals', {
        params: { page: pagination.current, page_size: pagination.pageSize },
      });
      setData(response.data.items || []);
      setPagination((prev) => ({ ...prev, total: response.data.total || 0 }));
    } catch (error) {
      message.error('获取信号数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [pagination.current, pagination.pageSize]);

  const handleConfirm = async (record: SignalRecord) => {
    try {
      await axios.post(`/api/signals/${record.id}/confirm`);
      message.success('已确认');
      fetchData();
    } catch {
      message.error('确认失败');
    }
  };

  const handleSkip = async (record: SignalRecord) => {
    try {
      await axios.post(`/api/signals/${record.id}/skip`);
      message.success('已跳过');
      fetchData();
    } catch {
      message.error('操作失败');
    }
  };

  const handleBatchConfirm = async () => {
    if (selectedRowKeys.length === 0) {
      message.warning('请先选择要确认的信号');
      return;
    }
    try {
      await axios.post('/api/signals/batch-confirm', { ids: selectedRowKeys });
      message.success(`已确认 ${selectedRowKeys.length} 条信号`);
      setSelectedRowKeys([]);
      fetchData();
    } catch {
      message.error('批量确认失败');
    }
  };

  const handleBatchSkip = async () => {
    if (selectedRowKeys.length === 0) {
      message.warning('请先选择要跳过的信号');
      return;
    }
    try {
      await axios.post('/api/signals/batch-skip', { ids: selectedRowKeys });
      message.success(`已跳过 ${selectedRowKeys.length} 条信号`);
      setSelectedRowKeys([]);
      fetchData();
    } catch {
      message.error('批量操作失败');
    }
  };

  const columns = [
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
      width: 120,
    },
    {
      title: '信号类型',
      dataIndex: 'signal_type',
      key: 'signal_type',
      width: 120,
      render: (type: string) => {
        const colorMap: Record<string, string> = {
          '突破买入': 'green',
          '止损卖出': 'red',
          '止盈卖出': 'orange',
          '趋势跟踪': 'blue',
        };
        return <Tag color={colorMap[type] || 'default'}>{type}</Tag>;
      },
    },
    {
      title: '信号时间',
      dataIndex: 'signal_time',
      key: 'signal_time',
      width: 180,
      render: (time: string) => dayjs(time).format('YYYY-MM-DD HH:mm:ss'),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      width: 100,
      render: (price: number) => price?.toFixed(2) ?? '-',
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => {
        const color = confidence >= 0.8 ? 'green' : confidence >= 0.5 ? 'orange' : 'red';
        return <Tag color={color}>{(confidence * 100).toFixed(0)}%</Tag>;
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          pending: 'gold',
          confirmed: 'green',
          skipped: 'gray',
        };
        const textMap: Record<string, string> = {
          pending: '待确认',
          confirmed: '已确认',
          skipped: '已跳过',
        };
        return <Tag color={colorMap[status]}>{textMap[status]}</Tag>;
      },
    },
    {
      title: '确认时间',
      dataIndex: 'confirmed_at',
      key: 'confirmed_at',
      width: 180,
      render: (time: string) => (time ? dayjs(time).format('YYYY-MM-DD HH:mm:ss') : '-'),
    },
    {
      title: '操作',
      key: 'action',
      width: 180,
      fixed: 'right' as const,
      render: (_: unknown, record: SignalRecord) => {
        if (record.status !== 'pending') {
          return <span style={{ color: '#999' }}>已处理</span>;
        }
        return (
          <Space>
            <Tooltip title="确认执行">
              <Button
                type="link"
                icon={<CheckCircleOutlined />}
                onClick={() => handleConfirm(record)}
                style={{ color: '#52c41a' }}
              />
            </Tooltip>
            <Tooltip title="跳过">
              <Button
                type="link"
                icon={<CloseCircleOutlined />}
                onClick={() => handleSkip(record)}
                style={{ color: '#ff4d4f' }}
              />
            </Tooltip>
            <Tooltip title="详情">
              <Button
                type="link"
                icon={<QuestionCircleOutlined />}
                onClick={() => Modal.info({ title: '信号详情', content: `股票: ${record.stock_name} (${record.stock_code})` })}
              />
            </Tooltip>
          </Space>
        );
      },
    },
  ];

  const rowSelection = {
    selectedRowKeys,
    onChange: (keys: React.Key[]) => setSelectedRowKeys(keys),
  };

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small" title="待处理信号">
            <h2 style={{ margin: 0 }}>{data.filter((d) => d.status === 'pending').length}</h2>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small" title="今日确认">
            <h2 style={{ margin: 0 }}>{data.filter((d) => d.status === 'confirmed').length}</h2>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small" title="今日跳过">
            <h2 style={{ margin: 0 }}>{data.filter((d) => d.status === 'skipped').length}</h2>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small" title="总计">
            <h2 style={{ margin: 0 }}>{pagination.total}</h2>
          </Card>
        </Col>
      </Row>

      <Space style={{ marginBottom: 16 }}>
        <Button type="primary" onClick={handleBatchConfirm} disabled={selectedRowKeys.length === 0}>
          批量确认 ({selectedRowKeys.length})
        </Button>
        <Button onClick={handleBatchSkip} disabled={selectedRowKeys.length === 0}>
          批量跳过 ({selectedRowKeys.length})
        </Button>
        <Button icon={<ReloadOutlined />} onClick={fetchData}>
          刷新
        </Button>
      </Space>

      <Table
        rowSelection={rowSelection}
        columns={columns}
        dataSource={data}
        loading={loading}
        rowKey="id"
        scroll={{ x: 1300 }}
        pagination={{
          current: pagination.current,
          pageSize: pagination.pageSize,
          total: pagination.total,
          showSizeChanger: true,
          showQuickJumper: true,
          showTotal: (total) => `共 ${total} 条`,
          onChange: (page, pageSize) => setPagination({ current: page, pageSize, total }),
        }}
      />
    </div>
  );
};

export default SignalTable;
