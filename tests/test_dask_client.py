import sys

sys.path.append("src/")

from dask_client import mapped_distance_matrix
from dask.distributed import Client
import numpy as np

client = Client(processes=False)


def identity(x):
    return x


def test_shape():
    pts = np.zeros((2, 2))

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([8, 8]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=1,
        func=np.sin,
        client=client,
    )
    assert m.shape == (8, 8)


def test_shape_nonsquare():
    pts = np.zeros((2, 2))

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([8, 20]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=1,
        func=np.sin,
        client=client,
    )
    assert m.shape == (8, 20)


def test_no_points():
    pts = np.zeros((0, 2))

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([8, 8]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=1,
        func=np.sin,
        client=client,
    )
    np.testing.assert_allclose(m, np.zeros((8, 8)))


def test_one_point_start():
    pts = np.array([[0.1, 0.1]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([8, 8]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.15,
        func=identity,
        client=client,
    )

    expected = np.zeros((8, 8), float)

    expected[0, 0] = np.sqrt(2 * 0.1 * 0.1)

    np.testing.assert_allclose(m, expected)


def test_one_point_end1():
    pts = np.array([[1.0, 1.0]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([4, 4]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.15,
        func=identity,
        client=client,
    )

    expected = np.zeros((4, 4), float)

    expected[-1, -1] = np.sqrt(2 * 0.1 * 0.1)

    np.testing.assert_allclose(m, expected)


def test_one_point_end2():
    pts = np.array([[2.2, 2.2]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([8, 8]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.15,
        func=identity,
        client=client,
    )

    expected = np.zeros((8, 8), float)

    expected[-1, -1] = np.sqrt(2 * 0.1 * 0.1)

    np.testing.assert_allclose(m, expected)


def test_one_point_end_periodic():
    pts = np.array([[2.3, 2.3]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([8, 8]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.15,
        func=identity,
        client=client,
    )

    expected = np.zeros((8, 8), float)

    expected[0, 0] = np.sqrt(2 * 0.1 * 0.1)

    np.testing.assert_allclose(m, expected)


def test_two_simple_points_nonsquare1():
    pts = np.array([[0.1, 3.8], [0.1, 2.0]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.2]),
        uniform_grid_size=np.array([8, 20]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.15,
        func=identity,
        client=client,
    )

    expected = np.zeros((8, 20), float)

    expected[0, [10, -1]] = 0.1

    np.testing.assert_allclose(m, expected)


def test_two_simple_points_nonsquare2():
    pts = np.array([[0.7, 3.79], [0.4, 2.0]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.2]),
        uniform_grid_size=np.array([8, 20]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.15,
        func=identity,
        client=client,
    )

    expected = np.zeros((8, 20), float)

    expected[2, -1] = np.sqrt(0.1 * 0.1 + 0.01 * 0.01)

    expected[1, 10] = 0.1

    np.testing.assert_allclose(m, expected)


def test_one_point_periodic1():
    pts = np.array([[0.1, 2.0]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.5]),
        uniform_grid_size=np.array([8, 4]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.41,
        func=identity,
        client=client,
    )

    expected = np.zeros((8, 4), float)

    expected[0, 0] = 0.1

    expected[-1, 0] = 0.4

    expected[1, 0] = 0.2

    np.testing.assert_allclose(m, expected)


def test_one_point_periodic2():
    pts = np.array([[0.1, 2.0]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.2]),
        uniform_grid_size=np.array([8, 10]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.41,
        func=identity,
        client=client,
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


def test_one_point_periodic3():
    pts = np.array([[0.1, 2.0]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.2]),
        uniform_grid_size=np.array([8, 10]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.45,
        func=identity,
        client=client,
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


def test_overlapping_points1():
    pts = np.array([[1.0, 1.0], [0.9, 0.91]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([4, 4]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.15,
        func=identity,
        client=client,
    )

    expected = np.zeros((4, 4), float)
    expected[-1, -1] = np.sqrt(2 * 0.1 * 0.1) + 0.01

    np.testing.assert_allclose(m, expected)


def test_overlapping_points2():
    pts = np.array([[1.0, 1.0], [0.9, 0.91], [0.89, 0.9]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([4, 4]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.15,
        func=identity,
        client=client,
    )

    expected = np.zeros((4, 4), float)
    expected[-1, -1] = np.sqrt(2 * 0.1 * 0.1) + 0.01 + 0.01

    np.testing.assert_allclose(m, expected)


def test_overlapping_points3():
    pts = np.array([[1.0, 1.0], [0.9, 0.91], [0.89, 0.9], [1.19, 1.2]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([4, 4]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.31,
        func=identity,
        client=client,
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


def test_overlapping_points_weight():
    pts = np.array([[1.0, 1.0], [0.9, 0.91], [0.89, 0.9], [1.19, 1.2]])

    m = mapped_distance_matrix(
        uniform_grid_cell_step=np.array([0.3, 0.3]),
        uniform_grid_size=np.array([4, 4]),
        bins_size=np.array([2, 2]),
        non_uniform_points=pts,
        max_distance=0.31,
        func=identity,
        client=client,
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
