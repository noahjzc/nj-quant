// web/frontend/src/pages/CronTracker/index.tsx
import React, { useEffect, useState } from 'react';
import { Card, Table, Tag, Descriptions, Space, Button } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, SyncOutlined } from '@ant-design/icons';
import axios from 'axios';

const STATUS_ICON: Record<string, React.ReactNode> = {
  success: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
  failed: <CloseCircleOutlined style={{ color: '#ff4d4f' }} />,
  running: <SyncOutlined spin style={{ color: '#1677ff' }} />,
};

const CronTracker: React.FC = () => {
  const [logs, setLogs] = useState([]);
  const [completeness, setCompleteness] = useState<any>({});
  const [loading, setLoading] = useState(false);

  const fetch = async () => {
    setLoading(true);
    try {
      const [logRes, statusRes] = await Promise.all([
        axios.get('/api/cron/'),
        axios.get('/api/cron/status'),
      ]);
      setLogs(logRes.data);
      setCompleteness(statusRes.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetch(); }, []);

  return (
    <div>
      <Card title="数据完整性" style={{ marginBottom: 24 }}>
        <Descriptions column={3}>
          <Descriptions.Item label="最近日线数据">{completeness.last_daily_date || '-'}</Descriptions.Item>
          <Descriptions.Item label="最近信号日期">{completeness.last_signal_date || '-'}</Descriptions.Item>
          <Descriptions.Item label="最近补全状态">
            {completeness.last_backfill?.status ? (
              <Tag icon={STATUS_ICON[completeness.last_backfill.status]}>
                {completeness.last_backfill.status} ({completeness.last_backfill.finished_at})
              </Tag>
            ) : '-'}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card
        title="任务执行记录"
        extra={<Button onClick={fetch} loading={loading}>刷新</Button>}
      >
        <Table
          dataSource={logs} rowKey="id" pagination={{ pageSize: 15 }}
          columns={[
            { title: '任务', dataIndex: 'task_name', width: 150 },
            {
              title: '状态', dataIndex: 'status', width: 100,
              render: (s: string) => <Tag icon={STATUS_ICON[s]} color={s === 'success' ? 'success' : s === 'failed' ? 'error' : 'processing'}>{s}</Tag>,
            },
            { title: '开始', dataIndex: 'started_at', width: 170 },
            { title: '结束', dataIndex: 'finished_at', width: 170, render: (v: string) => v || '-' },
            { title: '错误', dataIndex: 'error_message', ellipsis: true, render: (v: string) => v || '-' },
            { title: '详情', dataIndex: 'metadata', ellipsis: true, render: (v: any) => typeof v === 'string' ? v : JSON.stringify(v) || '-' },
          ]}
        />
      </Card>
    </div>
  );
};

export default CronTracker;