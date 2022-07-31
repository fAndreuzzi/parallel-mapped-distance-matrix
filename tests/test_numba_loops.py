import sys

sys.path.append("src/")

from numba_loops import mapped_distance_matrix
import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor, Executor, Future

import pytest


class FakeExecutor(Executor):
    def submit(self, f, *args, **kwargs):
        future = Future()
        future.set_result(f(*args, **kwargs))
        return future

    def shutdown(self, wait=True):
        pass


@nb.njit
def identity(x):
    return x


fake_pool_init = lambda: FakeExecutor()
thread_pool2_init = lambda: ThreadPoolExecutor(2)
thread_pool4_init = lambda: ThreadPoolExecutor(4)
thread_pool_initializers = (
    fake_pool_init,
    thread_pool2_init,
    thread_pool4_init,
)

# -------


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_shape(pool_init):
    pts = np.zeros((2, 2))

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([8, 8]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=1,
            func=identity,
            executor=executor,
        )
    assert m.shape == (8, 8)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_shape_nonsquare(pool_init):
    pts = np.zeros((2, 2))

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([8, 20]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=1,
            func=identity,
            executor=executor,
        )
    assert m.shape == (8, 20)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_no_points(pool_init):
    pts = np.zeros((0, 2))

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([8, 8]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=1,
            func=identity,
            executor=executor,
        )
    np.testing.assert_allclose(m, np.zeros((8, 8)))


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_one_point_start(pool_init):
    pts = np.array([[0.1, 0.1]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([8, 8]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.15,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((8, 8), float)

    expected[0, 0] = np.sqrt(2 * 0.1 * 0.1)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_one_point_end1(pool_init):
    pts = np.array([[1.0, 1.0]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([4, 4]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.15,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((4, 4), float)

    expected[-1, -1] = np.sqrt(2 * 0.1 * 0.1)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_one_point_end2(pool_init):
    pts = np.array([[2.2, 2.2]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([8, 8]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.15,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((8, 8), float)

    expected[-1, -1] = np.sqrt(2 * 0.1 * 0.1)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_one_point_end_periodic(pool_init):
    pts = np.array([[2.3, 2.3]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([8, 8]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.15,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((8, 8), float)

    expected[0, 0] = np.sqrt(2 * 0.1 * 0.1)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_two_simple_points_nonsquare1(pool_init):
    pts = np.array([[0.1, 3.8], [0.1, 2.0]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.2]),
            uniform_grid_size=np.array([8, 20]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.15,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((8, 20), float)

    expected[0, [10, -1]] = 0.1

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_two_simple_points_nonsquare2(pool_init):
    pts = np.array([[0.7, 3.79], [0.4, 2.0]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.2]),
            uniform_grid_size=np.array([8, 20]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.15,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((8, 20), float)

    expected[2, -1] = np.sqrt(0.1 * 0.1 + 0.01 * 0.01)

    expected[1, 10] = 0.1

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_one_point_periodic1(pool_init):
    pts = np.array([[0.1, 2.0]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.5]),
            uniform_grid_size=np.array([8, 4]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.41,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((8, 4), float)

    expected[0, 0] = 0.1

    expected[-1, 0] = 0.4

    expected[1, 0] = 0.2

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_one_point_periodic2(pool_init):
    pts = np.array([[0.1, 2.0]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.2]),
            uniform_grid_size=np.array([8, 10]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.41,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((8, 10), float)

    expected[0, 0] = 0.1

    expected[1, 0] = 0.2

    expected[-1, 0] = 0.4

    expected[0, 1] = np.sqrt(0.1 * 0.1 + 0.2 * 0.2)

    expected[0, -1] = np.sqrt(0.1 * 0.1 + 0.2 * 0.2)

    expected[1, 1] = np.sqrt(2 * 0.2 * 0.2)

    expected[1, -1] = np.sqrt(2 * 0.2 * 0.2)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_one_point_periodic3(pool_init):
    pts = np.array([[0.1, 2.0]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.2]),
            uniform_grid_size=np.array([8, 10]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.45,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((8, 10), float)

    expected[0, 0] = 0.1

    expected[1, 0] = 0.2

    expected[-1, 0] = 0.4

    expected[0, [-1, 1]] = np.sqrt(0.1 * 0.1 + 0.2 * 0.2)

    expected[1, [-1, 1]] = np.sqrt(2 * 0.2 * 0.2)

    expected[-1, [-1, 1]] = np.sqrt(0.4 * 0.4 + 0.2 * 0.2)

    expected[0, [2, -2]] = np.sqrt(0.1 * 0.1 + 0.4 * 0.4)

    expected[1, [2, -2]] = np.sqrt(0.2 * 0.2 + 0.4 * 0.4)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_overlapping_points1(pool_init):
    pts = np.array([[1.0, 1.0], [0.9, 0.91]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([4, 4]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.15,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((4, 4), float)
    expected[-1, -1] = np.sqrt(2 * 0.1 * 0.1) + 0.01

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_overlapping_points2(pool_init):
    pts = np.array([[1.0, 1.0], [0.9, 0.91], [0.89, 0.9]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([4, 4]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.15,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((4, 4), float)
    expected[-1, -1] = np.sqrt(2 * 0.1 * 0.1) + 0.01 + 0.01

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_overlapping_points3(pool_init):
    pts = np.array([[1.0, 1.0], [0.9, 0.91], [0.89, 0.9], [1.19, 1.2]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([4, 4]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.31,
            func=identity,
            executor=executor,
        )

    expected = np.zeros((4, 4), float)

    # 1
    expected[-1, -1] += np.sqrt(2 * 0.1 * 0.1)
    expected[0, -1] += np.sqrt(0.2 * 0.2 + 0.1 * 0.1)
    expected[-1, 0] += np.sqrt(0.2 * 0.2 + 0.1 * 0.1)
    expected[0, 0] += np.sqrt(0.2 * 0.2 * 2)

    # 2
    expected[-1, -1] += 0.01
    expected[-1, 0] += 0.29
    expected[[0, -2], -1] += np.sqrt(0.3 * 0.3 + 0.01 * 0.01)

    # 3
    expected[-1, -1] += 0.01
    expected[-2, -1] += 0.29
    expected[-1, [0, -2]] += np.sqrt(0.3 * 0.3 + 0.01 * 0.01)
    expected[0, -1] += 0.31

    # 4
    expected[0, 0] += 0.01
    expected[-1, 0] += 0.29
    expected[0, [-1, 1]] += np.sqrt(0.3 * 0.3 + 0.01 * 0.01)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_overlapping_points_weight(pool_init):
    pts = np.array([[1.0, 1.0], [0.9, 0.91], [0.89, 0.9], [1.19, 1.2]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            uniform_grid_cell_step=np.array([0.3, 0.3]),
            uniform_grid_size=np.array([4, 4]),
            bins_size=np.array([2, 2]),
            non_uniform_points=pts,
            max_distance=0.31,
            func=identity,
            executor=executor,
            weights=np.array([0.5, 0.2, 0.3, 0.4]),
        )

    expected = np.zeros((4, 4), float)

    # 1
    expected[-1, -1] += 0.5 * np.sqrt(2 * 0.1 * 0.1)
    expected[0, -1] += 0.5 * np.sqrt(0.2 * 0.2 + 0.1 * 0.1)
    expected[-1, 0] += 0.5 * np.sqrt(0.2 * 0.2 + 0.1 * 0.1)
    expected[0, 0] += 0.5 * np.sqrt(0.2 * 0.2 * 2)

    # 2
    expected[-1, -1] += 0.2 * 0.01
    expected[-1, 0] += 0.2 * 0.29
    expected[[0, -2], -1] += 0.2 * np.sqrt(0.3 * 0.3 + 0.01 * 0.01)

    # 3
    expected[-1, -1] += 0.3 * 0.01
    expected[-2, -1] += 0.3 * 0.29
    expected[-1, [0, -2]] += 0.3 * np.sqrt(0.3 * 0.3 + 0.01 * 0.01)
    expected[0, -1] += 0.3 * 0.31

    # 4
    expected[0, 0] += 0.4 * 0.01
    expected[-1, 0] += 0.4 * 0.29
    expected[0, [-1, 1]] += 0.4 * np.sqrt(0.3 * 0.3 + 0.01 * 0.01)

    np.testing.assert_allclose(m, expected)
