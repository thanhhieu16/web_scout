from app.usage import UsageCollector


def test_add_accumulates():
    u = UsageCollector()
    u.add(tokens=100, cost=0.01, searches=1)
    u.add(tokens=50, cost=0.005, searches=2)
    assert u.drain() == (150, 0.015, 3)


def test_drain_resets():
    u = UsageCollector()
    u.add(tokens=10, cost=0.1, searches=1)
    u.drain()
    assert u.drain() == (0, 0.0, 0)


def test_empty_collector_drains_to_zeros():
    assert UsageCollector().drain() == (0, 0.0, 0)
