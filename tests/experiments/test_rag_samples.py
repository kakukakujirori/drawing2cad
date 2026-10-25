from experiments.rag.samples import draw


def test_a_draw_leaves_out_test_parts_and_grows_without_reshuffling() -> None:
    pool = [f"{n:04d}" for n in range(100)]
    small, large = draw(pool, 10, {"0003"}), draw(pool, 30, {"0003"})
    assert "0003" not in large
    assert large[:10] == small
