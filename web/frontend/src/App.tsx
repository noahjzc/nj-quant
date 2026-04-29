import React from 'react';
import { BrowserRouter, Routes, Route, MenuProps } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import { TableOutlined, ShoppingOutlined, DatabaseOutlined, ClockCircleOutlined } from '@ant-design/icons';
import SignalTable from './pages/SignalTable';

const { Header, Content, Footer } = Layout;

const items: MenuProps['items'] = [
  { key: '/signal-table', icon: <TableOutlined />, label: '信号看板' },
  { key: '/positions', icon: <ShoppingOutlined />, label: '持仓管理' },
  { key: '/data-browser', icon: <DatabaseOutlined />, label: '数据浏览' },
  { key: '/cron-tracker', icon: <ClockCircleOutlined />, label: '定时任务' },
];

const App: React.FC = () => {
  return (
    <Layout className="layout" style={{ minHeight: '100vh' }}>
      <Header>
        <div className="logo" style={{ float: 'left', color: 'white', fontSize: '18px', marginRight: '40px' }}>
          NJ Quant 量化交易系统
        </div>
        <Menu
          theme="dark"
          mode="horizontal"
          defaultSelectedKeys={['/signal-table']}
          items={items}
          style={{ float: 'left' }}
        />
      </Header>
      <Content style={{ padding: '20px 50px' }}>
        <div className="site-layout-content">
          <Routes>
            <Route path="/signal-table" element={<SignalTable />} />
            <Route path="/positions" element={<div>持仓管理</div>} />
            <Route path="/data-browser" element={<div>数据浏览</div>} />
            <Route path="/cron-tracker" element={<div>定时任务</div>} />
          </Routes>
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}>NJ Quant 量化交易系统 ©2024</Footer>
    </Layout>
  );
};

export default App;
