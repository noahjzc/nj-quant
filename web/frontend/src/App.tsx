import React from 'react';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import { TableOutlined, ShoppingOutlined, DatabaseOutlined, ClockCircleOutlined } from '@ant-design/icons';
import SignalTable from './pages/SignalTable';
import Positions from './pages/Positions';
import DataBrowser from './pages/DataBrowser';
import CronTracker from './pages/CronTracker';

const { Header, Content, Footer } = Layout;

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Layout className="layout" style={{ minHeight: '100vh' }}>
        <Header>
          <div className="logo" style={{ float: 'left', color: 'white', fontSize: '18px', marginRight: '40px' }}>
            NJ Quant 量化交易系统
          </div>
          <MenuWithNavigate />
        </Header>
        <Content style={{ padding: '20px 50px' }}>
          <div className="site-layout-content">
            <Routes>
              <Route path="/signal-table" element={<SignalTable />} />
              <Route path="/positions" element={<Positions />} />
              <Route path="/data-browser" element={<DataBrowser />} />
              <Route path="/cron-tracker" element={<CronTracker />} />
            </Routes>
          </div>
        </Content>
        <Footer style={{ textAlign: 'center' }}>NJ Quant 量化交易系统 ©2024</Footer>
      </Layout>
    </BrowserRouter>
  );
};

const MenuWithNavigate: React.FC = () => {
  const navigate = useNavigate();
  const items = [
    { key: '/signal-table', icon: <TableOutlined />, label: '信号看板' },
    { key: '/positions', icon: <ShoppingOutlined />, label: '持仓管理' },
    { key: '/data-browser', icon: <DatabaseOutlined />, label: '数据浏览' },
    { key: '/cron-tracker', icon: <ClockCircleOutlined />, label: '定时任务' },
  ];
  return (
    <Menu
      theme="dark"
      mode="horizontal"
      defaultSelectedKeys={['/signal-table']}
      items={items}
      style={{ float: 'left' }}
      onClick={({ key }) => navigate(key)}
    />
  );
};

export default App;
