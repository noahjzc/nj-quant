"""历史初始化脚本测试"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from back_testing.data.sync.init_history import HistoryInitializer


class TestHistoryInitializer:
    """测试 HistoryInitializer"""

    @pytest.fixture
    def mock_initializer(self):
        with patch('back_testing.data.sync.init_history.get_engine'), \
             patch('back_testing.data.sync.init_history.get_session'):
            client = Mock()
            initializer = HistoryInitializer(client)
            return initializer

    def test_init_stock_meta_empty(self, mock_initializer):
        """测试空股票列表处理"""
        with patch.object(mock_initializer.client, 'get_stock_list', return_value=Mock(empty=True)):
            result = mock_initializer.init_stock_meta()
            assert result == 0