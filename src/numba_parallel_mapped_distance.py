import numpy as np
import numba as nb
from concurrent.futures import wait

from numpy_dimensional_utils import (
    add_to_slice,
    extract_slice,
    periodic_inner_sum,
)

from parallel_mapped_distance import (
    group_by,
    extract_subproblems,
    generate_uniform_grid,
)

add_to_slice = nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)(
    add_to_slice
)


def start_subproblem(
    bin_content,
    pts,
    bins_coords,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_mapped_distance_matrix,
):
    compute_mapped_distance_on_subgroup(
        subgroup_content=pts[bin_content],
        bin_coords=bins_coords[bin_content[0]],
        nup_idxes=bin_content,
        weights=weights[bin_content],
        uniform_grid_cell_step=uniform_grid_cell_step,
        bins_size=bins_size,
        max_distance=max_distance,
        max_distance_in_cells=max_distance_in_cells,
        function=function,
        reference_bin=reference_bin,
        exact_max_distance=exact_max_distance,
        global_mapped_distance_matrix=global_mapped_distance_matrix,
    )


def distribute_and_start_subproblems(
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    pts,
    weights,
    pts_per_future,
    executor,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_mapped_distance_matrix,
):
    r"""
    Given a set of points and the features of the uniform grid, find the
    appopriate bin for each point in `pts`, and split each bin in subproblems
    according to the given value of `pts_per_future`. The given `Client`
    instance is used to send each subproblem to the appropriate worker (no
    computation is started though).

    Returns
    -------
    `iterable`
    An iterable of Dask Future, one for each subproblem. Each Future encloses a
    tuple which contains three values:

        1. Points in the subproblem (a 2D NumPy array);
        2. (Non-linearized) coords of the bin which contains the subproblem;
        3. Indexes of the non-uniform points in this subproblem wrt `pts`;
        4. Weights for the non uniform points in this subproblem.
    """
    bins_per_axis = uniform_grid_size // bins_size

    # periodicity
    pts = np.mod(pts, (uniform_grid_size * uniform_grid_cell_step)[None])

    pts_bin_coords = np.floor_divide(
        pts, uniform_grid_cell_step * bins_size
    ).astype(int)

    # transform the N-Dimensional bins indexing (N is the number of axes)
    # into a linear one (only one index)
    linearized_bin_coords = np.ravel_multi_index(
        pts_bin_coords.T, bins_per_axis
    )
    # augment the linear indexing with the index of the point before using
    # group by, in order to have an index that we can use to access the
    # pts array
    aug_linearized_bin_coords = np.stack(
        (linearized_bin_coords, np.arange(len(pts))), axis=-1
    )
    indexes_inside_bins = group_by(aug_linearized_bin_coords)

    # we create subproblems for each bin (i.e. we split points in the
    # same bin in order to treat at most pts_per_future points in each Future)
    subproblems = extract_subproblems(indexes_inside_bins, pts_per_future)
    # each subproblem is treated by a single Future. each bin spawns one or
    # more subproblems.

    return (
        executor.submit(
            start_subproblem,
            subgroup,
            pts,
            pts_bin_coords,
            weights,
            uniform_grid_cell_step,
            bins_size,
            max_distance,
            max_distance_in_cells,
            function,
            reference_bin,
            exact_max_distance,
            global_mapped_distance_matrix,
        )
        for bin_content in subproblems
        for subgroup in bin_content
    )


@nb.generated_jit(nopython=True)
def compute_mapped_distance_on_subgroup(
    subgroup_content,
    bin_coords,
    nup_idxes,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_mapped_distance_matrix,
):
    if exact_max_distance:
        return compute_mapped_distance_on_subgroup_exact_distance
    else:
        return compute_mapped_distance_on_subgroup_nexact_distance


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True, parallel=True)
def compute_mapped_distance_on_subgroup_nexact_distance(
    subgroup_content,
    bin_coords,
    nup_idxes,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_mapped_distance_matrix,
):
    r"""
    Function to be executed on the worker, provides a mapping for each
    subproblem which computes the mapped distance.

    Returns
    -------
    `tuple`
    """

    M, N, D = reference_bin.shape
    L = len(subgroup_content)

    bin_virtual_lower_left = np.zeros(D)
    bin_virtual_upper_right = np.zeros(D)
    for i in range(D):
        bin_virtual_lower_left[i] = bin_coords[i] * bins_size[i]
        bin_virtual_upper_right[i] = (
            bin_virtual_lower_left[i] + bins_size[i] - 1
        )

    offset = -(bin_virtual_lower_left * uniform_grid_cell_step)

    mapped_distance = np.zeros((M, N))
    mapped_distances_1d = np.zeros(L)
    for j in range(M):
        for k in range(N):
            uniform_point = reference_bin[j, k]
            for i in nb.prange(L):
                translated_nup = subgroup_content[i] + offset
                d = np.linalg.norm(uniform_point - translated_nup)
                mapped_distances_1d[i] = function(d) * weights[i]

            mapped_distance[j, k] = np.sum(mapped_distances_1d)
            mapped_distances_1d = np.zeros(L)

    add_to_slice(
        global_mapped_distance_matrix,
        mapped_distance,
        bin_virtual_lower_left,
        bin_virtual_upper_right + 1 + 2 * max_distance_in_cells,
    )


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True, parallel=True)
def compute_mapped_distance_on_subgroup_exact_distance(
    subgroup_content,
    bin_coords,
    nup_idxes,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_mapped_distance_matrix,
):
    r"""
    Function to be executed on the worker, provides a mapping for each
    subproblem which computes the mapped distance.

    Returns
    -------
    `tuple`
    """

    M, N, D = reference_bin.shape
    L = len(subgroup_content)

    bin_virtual_lower_left = np.zeros(D)
    bin_virtual_upper_right = np.zeros(D)
    for i in range(D):
        bin_virtual_lower_left[i] = bin_coords[i] * bins_size[i]
        bin_virtual_upper_right[i] = (
            bin_virtual_lower_left[i] + bins_size[i] - 1
        )

    offset = -(bin_virtual_lower_left * uniform_grid_cell_step)

    mapped_distance = np.zeros((M, N))
    mapped_distances_1d = np.zeros(L)
    for j in range(M):
        for k in range(N):
            uniform_point = reference_bin[j, k]
            for i in nb.prange(L):
                translated_nup = subgroup_content[i] + offset
                d = np.linalg.norm(uniform_point - translated_nup)
                if d < max_distance:
                    mapped_distances_1d[i] = function(d) * weights[i]

            mapped_distance[j, k] = np.sum(mapped_distances_1d)
            mapped_distances_1d = np.zeros(L)

    add_to_slice(
        global_mapped_distance_matrix,
        mapped_distance,
        bin_virtual_lower_left,
        bin_virtual_upper_right + 1 + 2 * max_distance_in_cells,
    )


def mapped_distance_matrix(
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    non_uniform_points,
    max_distance,
    func,
    executor,
    weights=None,
    exact_max_distance=True,
    pts_per_future=5,
    cell_reference_point_offset=0,
):
    r"""
    Compute the mapped distance matrix of a set of non uniform points
    distributed inside a uniform grid whose features are specified in
    `uniform_grid_cell_step`, `uniform_grid_size`, `bins_size`.

    Returns
    -------
    `np.ndarray`
    """
    if weights is None:
        weights = np.ones(len(non_uniform_points), dtype=int)
    dtype = non_uniform_points.dtype
    if len(non_uniform_points) == 0:
        return np.zeros(uniform_grid_size, dtype=dtype)

    # bins should divide properly the grid
    assert np.all(np.mod(uniform_grid_size, bins_size) == 0)

    max_distance_in_cells = np.ceil(
        max_distance / uniform_grid_cell_step
    ).astype(int)

    # build a reference padded bin. the padding is given by taking twice
    # (in each direction) the value of max_distance. the lower_left point is
    # set to max_distance because we want the (0,0) point to be the first
    # point inside the non-padded bin.
    reference_bin = generate_uniform_grid(
        uniform_grid_cell_step, bins_size + 2 * max_distance_in_cells
    )
    lower_left = -(max_distance_in_cells * uniform_grid_cell_step)
    reference_bin += lower_left + cell_reference_point_offset

    mapped_distance = np.zeros(
        uniform_grid_size + 2 * max_distance_in_cells,
        dtype=dtype,
    )

    # split and distribute subproblems to the workers
    futures = distribute_and_start_subproblems(
        uniform_grid_cell_step=uniform_grid_cell_step,
        uniform_grid_size=uniform_grid_size,
        bins_size=bins_size,
        pts=non_uniform_points,
        pts_per_future=pts_per_future,
        executor=executor,
        weights=weights,
        max_distance=max_distance,
        max_distance_in_cells=max_distance_in_cells,
        reference_bin=reference_bin,
        function=func,
        exact_max_distance=exact_max_distance,
        global_mapped_distance_matrix=mapped_distance,
    )

    wait(tuple(futures))

    return periodic_inner_sum(
        mapped_distance,
        max_distance_in_cells,
        uniform_grid_size + max_distance_in_cells,
    )
