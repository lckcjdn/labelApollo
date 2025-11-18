import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import MainInterface from './components/MainInterface';
import './App.css';

const { Header, Content } = Layout;

const App: React.FC = () => {
  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ background: '#001529', padding: '0 20px' }}>
          <h1 style={{ color: 'white', margin: 0 }}>Label Apollo - AI图像标注工具</h1>
        </Header>
        <Content style={{ padding: '20px' }}>
          <Routes>
            <Route path="/" element={<MainInterface />} />
          </Routes>
        </Content>
      </Layout>
    </Router>
  );
};

export default App;