import pytest


@pytest.fixture(autouse=True, scope="session")
def _cache_outside_the_home_directory(tmp_path_factory):
    """Comparisons keep what they read in the user's cache; tests start empty."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("cache")))
        yield
