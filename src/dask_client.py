import numpy as np

from dask.distributed import as_completed

from numpy_dimensional_utils import (
    add_to_slice,
    extract_slice,
    periodic_inner_sum,
)


def group_by(a):
    r"""
    Groups the values in `a[:,1]` according to the values in `a[:,0]`. Produces
    a list of lists, each inner list is the set of values in `a[:,1]` that
    share a common value in the first column.

    Parameters
    ----------
    a: np.ndarray
        2D NumPy array, should have two columns. The first one contains the
        reference values to be used for the grouping, the second should
        contain the values to be grouped.

    Returns
    -------
    `list`
    A list of grouped values from the second column of `a`, according to the
    first column of `a`.

    Example
    -------
    The expected returned value for:

        >>> bin_coords = [
        ...     [0, 1],
        ...     [1, 2],
        ...     [1, 3],
        ...     [0, 4],
        ...     [2, 5]
        >>> ]

    is `[[1,4], [2,3], [5]]`.
    """

    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


def extract_subproblems(indexes, n_per_subgroup):
    r"""
    Given a set of indexes grouped by bin, extract subproblems from each bin
    according to the parameter `n_per_subgroup`.

    Parameters
    ----------
    indexes: list
        `list` of list. Each inner list contains the set of indexes inside
        the corresponding bin.
    n_per_subgroup: int
        Number of points in a subproblem. If `-1`, then there's no upper bound.

    Returns
    -------
    `iterable`
    An iterable whose elements correspond to bins. Each iterable wraps an
    iterable of subproblems.
    """
    if n_per_subgroup != -1:
        return map(
            # here we apply the finer granularity (#pts per future)
            lambda arr: np.array_split(
                arr, np.ceil(len(arr) / n_per_subgroup)
            ),
            indexes,
        )
    else:
        # we transform the lists of indexes in indexes_inside_bins to NumPy
        # arrays. we also wrap them into 1-element tuples because of how we
        # treat them in client.map
        return map(lambda arr: (np.array(arr),), indexes)


def distribute_subproblems(
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    pts,
    weights,
    pts_per_future,
    client,
    scatter,
):
    r"""
    Given a set of points and the features of the uniform grid, find the
    appopriate bin for each point in `pts`, and split each bin in subproblems
    according to the given value of `pts_per_future`. The given `Client`
    instance is used to send each subproblem to the appropriate worker (no
    computation is started though).

    Parameters
    ----------
    uniform_grid_cell_step: np.ndarray
        Dimension of cells of the uniform grid.
    uniform_grid_size: np.ndarray
        Number of cells in the uniform grid (for each axis). Expected a 1D NumPy
        array whose length is the number of dimensions of the space.
    bins_size: np.ndarray
        Number of cells in a bin (for each axis). Expected a NumPy
        1D array whose length is the number of dimensions of the space.
    pts: np.ndarray
        Non-uniform points scattered in the uniform grid. Expected a 2D NumPy
        array whose number of rows is the number of points, and whose number of
        columns is the number of dimensions of the space.
    weights: np.ndarray
        Weights defined as for the parameter `weights` in
        :func:`mapped_distance_matrix`.
    pts_per_future: int
        Number of points in a subproblem. If `-1`, then there's no upper bound.
    client: dask.distributed.Client
        Dask `Client` to be used to move resources to the appropriate places in
        order to have them ready for the following computation.
    scatter: boolean
        If true, data is "scattered" (i.e. `Client.scatter`) before starting
        the computation.

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

    if scatter:
        bin_coords_fu = client.scatter(pts_bin_coords, broadcast=True)
        pts_fu = client.scatter(pts, broadcast=True)
    else:
        bin_coords_fu = pts_bin_coords
        pts_fu = pts

    def bin_content_tuple(bin_content, pts, pts_bin_coords):
        return (
            pts[bin_content],
            pts_bin_coords[bin_content[0]],
            bin_content,
            weights[bin_content],
        )

    return client.map(
        bin_content_tuple,
        tuple(
            subgroup for bin_content in subproblems for subgroup in bin_content
        ),
        pts=pts_fu,
        pts_bin_coords=bin_coords_fu,
    )


def compute_mapped_distance_on_subgroup(
    subgroup_info,
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    exact_max_distance,
    reference_bin,
    dtype,
):
    r"""
    Function to be executed on the worker, provides a mapping for each
    subproblem which computes the mapped distance.

    Returns
    -------
    `tuple`
    """
    subgroup, bin_coords, nup_idxes, weights = subgroup_info

    # location of the lower left point of the non-padded bin in terms
    # of uniform grid cells
    bin_virtual_lower_left = bin_coords * bins_size
    # location of the upper right point of the non-padded bin
    bin_virtual_upper_right = bin_virtual_lower_left + bins_size - 1

    # translate the subgroup in order to locate it nearby the reference bin
    # (reminder: the lower left point of the (non-padded) reference bin is
    # [0,0]).
    subgroup -= bin_virtual_lower_left * uniform_grid_cell_step

    distances = np.linalg.norm(
        reference_bin[:, :, None] - subgroup[None, None],
        axis=-1,
    )

    if exact_max_distance:
        nearby = distances <= max_distance
        mapped_distance = np.zeros_like(distances, dtype=dtype)
        mapped_distance[nearby] = function(distances[nearby])
    else:
        mapped_distance = function(distances)

    # we add one because the upper bound is not included
    bin_bounds = np.array(
        [bin_virtual_lower_left, bin_virtual_upper_right + 1]
    )
    bin_bounds[1] += 2 * max_distance_in_cells

    return mapped_distance.dot(weights), bin_bounds


def generate_uniform_grid(grid_step, grid_size):
    r"""
    Generate an uniform grid according to the given features in `D` dimensions.

    Parameters
    ----------
    grid_step: np.ndarray
        Size of the step of the grid in each direction. Expected a 1D NumPy
        array whose size is `D`.
    grid_size: np.ndarray
        Number of cells in the grid in each direction. Expected a 1D NumPy
        array whose size is `D`.

    Returns
    -------
    `np.ndarray`
    """
    # the tranpose is needed because we want the components in the rightmost
    # part of the shape
    integer_grid = extract_slice(
        np.mgrid, np.zeros_like(grid_size), grid_size
    ).T
    # TODO 3d
    grid = integer_grid * grid_step[None, None]

    return np.swapaxes(grid, 0, 1)


def mapped_distance_matrix(
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    non_uniform_points,
    max_distance,
    func,
    client,
    weights=None,
    exact_max_distance=True,
    pts_per_future=5,
    dtype=None,
    cell_reference_point_offset=0,
    scatter=True,
):
    r"""
    Compute the mapped distance matrix of a set of non uniform points
    distributed inside a uniform grid whose features are specified in
    `uniform_grid_cell_step`, `uniform_grid_size`, `bins_size`.

    Parameters
    ----------
    uniform_grid_cell_step: np.ndarray
        Dimension of cells of the uniform grid.
    uniform_grid_size: np.ndarray
        Number of cells in the uniform grid (for each axis). Expected a 1D NumPy
        array whose length is the number of dimensions of the space.
    bins_size: np.ndarray
        Number of cells in a bin (for each axis). Expected a NumPy
        1D array whose length is the number of dimensions of the space.
    non_uniform_points: np.ndarray
        Non-uniform points scattered in the uniform grid. Expected a 2D NumPy
        array whose number of rows is the number of points, and whose number of
        columns is the number of dimensions of the space.
    max_distance: int
        Maximum distance between a pair uniform/non-uniform point to be
        considered not zero.
    function: function
        Function used to map the distance between uniform and non-uniform
        points.
    client: dask.distributed.Client
        Dask `Client` used for the computations.
    weights: np.ndarray
        Weights used to scale the contribution of each non-uniform point into
        the uniform grid.
    exact_max_distance: bool
        If `True`, the result is more precise but might require some additional
        work and thus damage performance.
        # TODO explain better
    pts_per_future: int
        Number of points in a subproblem. If `pts_per_future=-1`, then there's
        no upper bound.
    cell_reference_point_offset: np.array or int
        Offset of the reference point (i.e. the point used to compute the
        distance between a cell and a non-uniform point) from the bottom left
        point of the cell. Zero by default, the reference point should never
        lie outside the cell to preserve the correctness of the algorithm. No
        sanity checks are performed.
    scatter: boolean
        If true, data is "scattered" (i.e. `Client.scatter`) before starting
        the computation.

    Returns
    -------
    `np.ndarray`
    """
    if weights is None:
        weights = np.ones(len(non_uniform_points), dtype=int)
    if dtype is None:
        dtype = non_uniform_points.dtype
    if len(non_uniform_points) == 0:
        return np.zeros(uniform_grid_size, dtype=dtype)

    # bins should divide properly the grid
    assert np.all(np.mod(uniform_grid_size, bins_size) == 0)

    # split and distribute subproblems to the workers
    subgroups_coords_fu = distribute_subproblems(
        uniform_grid_cell_step=uniform_grid_cell_step,
        uniform_grid_size=uniform_grid_size,
        bins_size=bins_size,
        pts=non_uniform_points,
        pts_per_future=pts_per_future,
        client=client,
        weights=weights,
        scatter=scatter,
    )

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

    # start computation of the mapped distance
    mapped_distances_fu = client.map(
        compute_mapped_distance_on_subgroup,
        subgroups_coords_fu,
        uniform_grid_size=uniform_grid_size,
        uniform_grid_cell_step=uniform_grid_cell_step,
        bins_size=bins_size,
        function=func,
        reference_bin=reference_bin,
        max_distance=max_distance,
        max_distance_in_cells=max_distance_in_cells,
        exact_max_distance=exact_max_distance,
        dtype=dtype,
    )

    mapped_distance = np.zeros(
        uniform_grid_size + 2 * max_distance_in_cells,
        dtype=dtype,
    )

    for _, (bin_aggregated_grid, bin_bounds) in as_completed(
        mapped_distances_fu, with_results=True
    ):
        add_to_slice(
            mapped_distance, bin_aggregated_grid, bin_bounds[0], bin_bounds[1]
        )

    return periodic_inner_sum(
        mapped_distance,
        max_distance_in_cells,
        uniform_grid_size + max_distance_in_cells,
    )
