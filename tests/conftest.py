import pytest

from workhouse.company import build_company
from workhouse.config import Settings
from workhouse.store import Store


@pytest.fixture
def settings(tmp_path):
    return Settings(mode="mock", db_path=tmp_path / "wh.db", operator_profile_path=tmp_path / "operator.json", tick_seconds=0.0, max_total_cost_usd=100.0)


@pytest.fixture
def engine(settings):
    return build_company(settings, store=Store(settings.db_path))
