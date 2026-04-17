import React, { useState } from 'react';
import { Layout, Menu, Button } from 'antd';
import {
    DashboardOutlined,
    UserOutlined,
    CloudServerOutlined,
    DatabaseOutlined,
    FileTextOutlined,
    ArrowLeftOutlined
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { AdminDashboard } from './components/AdminDashboard';
import { AdminUserTable } from './components/AdminUserTable';
import { AdminModelManagement } from './components/AdminModelManagement';
import { AdminDataManagement } from './components/AdminDataManagement';
import { AdminStrategyTemplates } from './components/AdminStrategyTemplates';
import { AdminTab } from './types';

const { Sider, Content } = Layout;

const AdminPage: React.FC = () => {
    const [activeTab, setActiveTab] = useState<AdminTab>('dashboard');
    const navigate = useNavigate();

    const renderContent = () => {
        switch (activeTab) {
            case 'dashboard': return <AdminDashboard />;
            case 'users': return <AdminUserTable />;
            case 'models': return <AdminModelManagement />;
            case 'data': return <AdminDataManagement />;
            case 'strategy-templates': return <AdminStrategyTemplates />;
            default: return <AdminDashboard />;
        }
    };

    const menuItems = [
        { key: 'dashboard', icon: <DashboardOutlined />, label: '系统概览' },
        { key: 'users', icon: <UserOutlined />, label: '用户管理' },
        { key: 'models', icon: <CloudServerOutlined />, label: '模型管理' },
        { key: 'data', icon: <DatabaseOutlined />, label: '数据管理' },
        { key: 'strategy-templates', icon: <FileTextOutlined />, label: '策略模板' },
    ];

    return (
        <Layout className="min-h-screen bg-white">
            <Sider
                width={240}
                theme="light"
                className="border-r border-slate-100"
                style={{
                    overflow: 'auto',
                    height: '100vh',
                    position: 'fixed',
                    left: 0,
                    top: 0,
                    bottom: 0,
                }}
            >
                <div className="p-6 flex items-center gap-3 mb-6">
                    <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-black text-white">Q</div>
                    <span className="font-bold text-slate-900 tracking-tight">Admin Console</span>
                </div>

                <Menu
                    mode="inline"
                    selectedKeys={[activeTab]}
                    items={menuItems}
                    onSelect={({ key }) => setActiveTab(key as AdminTab)}
                    className="border-none"
                />

                <div className="absolute bottom-8 left-0 w-full px-6">
                    <Button
                        icon={<ArrowLeftOutlined />}
                        block
                        className="rounded-xl border-slate-200 text-slate-500 font-bold"
                        onClick={() => navigate('/')}
                    >
                        返回工作台
                    </Button>
                </div>
            </Sider>

            <Layout style={{ marginLeft: 240, background: '#fff', height: '100vh', overflowY: 'auto' }}>
                <Content className="p-10 max-w-6xl mx-auto w-full" style={{ paddingBottom: 100 }}>
                    {activeTab === 'dashboard' && (
                        <header className="mb-10">
                            <h1 className="text-2xl font-black text-slate-800 uppercase tracking-tight">
                                {menuItems.find(m => m.key === activeTab)?.label}
                            </h1>
                            <p className="text-slate-400 text-sm mt-1">管理控制台 · QuantMind Enterprise v2.0</p>
                        </header>
                    )}

                    <main className="bg-white rounded-3xl">
                        {renderContent()}
                    </main>
                </Content>
            </Layout>
        </Layout>
    );
};

export default AdminPage;
