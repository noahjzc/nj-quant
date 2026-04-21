"""akshare_client 测试"""
import pytest
from datetime import date, timedelta
from back_testing.data.sync.akshare_client import AkshareClient


class TestAkshareClient:
    """测试 AkshareClient"""

    @pytest.fixture
    def client(self):
        return AkshareClient(rate_limit=1)

    def test_init(self):
        """测试初始化"""
        client = AkshareClient()
        assert client.rate_limit == 10
        assert client.retry_times == 3

    def test_rate_limit_wait(self, client):
        """测试限速等待"""
        import time
        start = time.time()
        client._rate_limit_wait()
        client._rate_limit_wait()
        elapsed = time.time() - start
        assert elapsed >= 0.09  # 至少等待 0.1 秒

    def test_get_stock_list(self, client):
        """测试获取股票列表"""
        df = client.get_stock_list()
        assert not df.empty
        assert 'stock_code' in df.columns
        assert 'stock_name' in df.columns
        # 验证格式
        for code in df['stock_code'].head():
            assert code.startswith('sh') or code.startswith('sz')
