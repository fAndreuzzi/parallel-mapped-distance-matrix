import numpy as np

from dask.distributed import as_completed


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


def compute_linearized_bin_coords(bins_per_axis, bins_coords):
    r"""
    Given a set of N-dimesional bin indexes (for each point we have N indexes,
    one for each axis) this function linearizes the coordinates in order to
    associate a unique index in an 1D array to each bin.

    Linear coordinates are assigned in C-like order (last axis changes faster).

    Parameters
    ----------
    bins_per_axis: np.ndarray
        1D NumPy array containing the number of bins for each axis.
    bins_coords: np.ndarray
        2D NumPy array containing the bin coordinates to be linearized. Each row
        is a bin coordinate, having a number of column equal to the number of
        dimensions of the space.

    Returns
    -------
    `np.ndarray`
    An 1D NumPy array which contains the linearized coordinates of the
    given bins (starting from zero).

    Example
    -------
    If `bins_per_axis = [1,3,2]` and:

        >>> bin_coords = [
        ...     [0, 1, 1],
        ...     [1, 1, 1],
        ...     [0, 0, 1],
        ...     [1, 2, 0]
        >>> ]

    then the expected value returned is `[3, 9, 1, 10]`.
    """
    shifted_nbins_per_axis = np.ones_like(bins_per_axis, dtype=int)
    shifted_nbins_per_axis[:-1] = bins_per_axis[1:]

    return np.dot(bin_coords, shifted_nbins_per_axis[:, None])


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
        # indexes inside bins splitted according to pts_per_future
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
    uniform_grid_cell_size,
    uniform_grid_cell_count,
    bins_size,
    pts,
    weights,
    pts_per_future,
    client,
):
    r"""
    Given a set of points and the features of the uniform grid, find the
    appopriate bin for each point in `pts`, and split each bin in subproblems
    according to the given value of `pts_per_future`. The given `Client`
    instance is used to send each subproblem to the appropriate worker (no
    computation is started though).

    Parameters
    ----------
    uniform_grid_cell_size: np.ndarray
        Dimension of a cell of the grid (for each axis). Expected a 1D NumPy
        array whose length is the number of dimensions of the space.
    uniform_grid_cell_count: np.ndarray
        Number of cells in the uniform grid (for each axis). Expected a 1D NumPy
        array whose length is the number of dimensions of the space.
    bins_size: np.ndarray
        Number of cells of the grid in a bin (for each axis). Expected a NumPy
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
    # number of bins per axis
    bins_per_axis = uniform_grid_cell_count // bins_size

    # for each point in pts, compute the coordinate in the uniform grid
    bin_coords = np.floor_divide(
        pts, uniform_grid_cell_size * bins_size
    ).astype(int)
    # moves to the last bin of the axis any point which is outside the region
    # defined by samples2.
    np.clip(bin_coords, None, uniform_grid_cell_count - 1, out=bin_coords)

    # we transform our N-Dimensional bins indexing (N is the number of axes)
    # into a linear one (only one index)
    linearized_bin_coords = compute_linearized_bin_coords(
        bins_per_axis, bin_coords
    )
    # we agument the linear indexing with the index of the point before using
    # group by, in order to have an index to use in order to access the
    # pts array
    aug_linearized_bin_coords = np.hstack(
        [linearized_bin_coords, np.arange(len(pts))[:, None]]
    )
    # group by puts into the same group those points which are in the same bin
    indexes_inside_bins = group_by(aug_linearized_bin_coords)

    # we create subproblems for each bin (i.e. we split points in the
    # same bin in order to treat at most pts_per_future points in each Future)
    subproblems = extract_subproblems(indexes_inside_bins, pts_per_future)
    # each subproblem is treated by a single Future. each bin spawns one or
    # more subproblems.

    # TODO needed?
    bin_coords_fu = client.scatter(bin_coords, broadcast=True)
    pts_fu = client.scatter(pts, broadcast=True)

    def bin_content_tuple(bin_content, pts, bin_coords):
        return (
            pts[bin_content],
            bin_coords[bin_content[0]],
            bin_content,
            weights[bin_content],
        )

    return client.map(
        bin_content_tuple,
        (subgroup for bin_content in subproblems for subgroup in bin_content),
        pts=pts_fu,
        bin_coords=bin_coords_fu,
    )


def build_uniform_grid_bin():
    return grid, idxes


def compute_mapped_distance_on_subgroup(
    subgroup_info,
    uniform_grid_cell_size,
    uniform_grid_cell_count,
    bins_size,
    max_distance_in_cells,
    function,
    exact_max_distance,
    reference_bin,
):
    r"""
    Function to be executed on the worker, provides a mapping for each
    subproblem which computes the mapped distance.

    Returns
    -------
    `tuple`
    """
    # unwrap the content of the future
    subgroup, bin_coords, nup_idxes, weights = subgroups_and_coords

    uniform_grid, samples1_idxes = build_uniform_grid_bin()

    distances = np.linalg.norm(
        uniform_grid[..., None] - subgroup.T[:, None, None],
        axis=0,
    )

    if exact_max_distance:
        nearby = distances < max_distance
        mapped_distance = np.zeros_like(distances, dtype=subgroup.dtype)
        mapped_distance[nearby] = function(distances[nearby])
    else:
        mapped_distance = function(distances)

    return mapped_distance, (*samples1_idxes, nup_idxes)


def mapped_distance_matrix(
    uniform_grid_cell_size,
    uniform_grid_cell_count,
    bins_size,
    non_uniform_points,
    max_distance,
    func,
    client,
    weights=None,
    exact_max_distance=True,
    pts_per_future=5,
):
    r"""
    Compute the mapped distance matrix of a set of non uniform points
    distributed inside a uniform grid whose features are specified in
    `uniform_grid_cell_size`, `uniform_grid_cell_count`, `bins_size`.

    Parameters
    ----------
    uniform_grid_cell_size: np.ndarray
        Dimension of a cell of the grid (for each axis). Expected a 1D NumPy
        array whose length is the number of dimensions of the space.
    uniform_grid_cell_count: np.ndarray
        Number of cells in the uniform grid (for each axis). Expected a 1D NumPy
        array whose length is the number of dimensions of the space.
    bins_size: np.ndarray
        Number of cells of the grid in a bin (for each axis). Expected a NumPy
        1D array whose length is the number of dimensions of the space.
    non_uniform_points: np.ndarray
        Non-uniform points scattered in the uniform grid. Expected a 2D NumPy
        array whose number of rows is the number of points, and whose number of
        columns is the number of dimensions of the space.
    max_distance: int
        Maximum distance between a pair uniform/non-uniform point to be
        considered not zero, in terms of cells of the uniform grid.
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

    Returns
    -------
    `np.ndarray`
    """
    if weights is None:
        weights = np.ones(len(non_uniform_points), dtype=int)

    # split and distribute subproblems to the workers
    subgroups_coords_fu = distribute_subproblems(
        uniform_grid_cell_size,
        uniform_grid_cell_count,
        samples2,
        bins_size,
        pts_per_future=pts_per_future,
        client=client,
    )

    mapped_distances_fu = client.map(
        compute_mapped_distance_on_subgroup,
        subgroups_coords_fu,
        uniform_grid_cell_count=uniform_grid_cell_count,
        uniform_grid_cell_size=uniform_grid_cell_size,
        bins_size=bins_size,
        function=func,
        weights=weights,
        max_distance=max_distance,
        exact_max_distance=exact_max_distance,
    )

    mapped_distance = np.zeros(
        (*uniform_grid_cell_count, len(samples2)),
        dtype=samples2.dtype,
    )
    for _, (submatrix, idxes_broadcastable_tuple) in as_completed(
        mapped_distances_fu, with_results=True
    ):
        mapped_distance[idxes_broadcastable_tuple] = submatrix

    return mapped_distance.reshape(-1, len(samples2))
