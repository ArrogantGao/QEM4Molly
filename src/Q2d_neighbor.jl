export Q2d_neighbor, update_neighbor

mutable struct Q2d_Cell_List{CL, NL, NS, IP}
    cell_list::CL
    neighbor_list::NL
    n_step::NS
    iter_period::IP
end

function Q2d_neighbor(x_coords::Vector, cutoff::Real, unitcell::Vector; n_step::Int = 0, iter_period::Int = 100)
    n_atoms = size(x_coords)[1]
    x_q2d = [[x_coords[i][1], x_coords[i][2]] for i in 1:n_atoms]
    cell_list = InPlaceNeighborList(x = x_q2d, cutoff = cutoff, unitcell = unitcell, parallel=false)
    neighbor_list = neighborlist!(cell_list)
    return Q2d_Cell_List{typeof(cell_list), typeof(neighbor_list), Int, Int}(cell_list, neighbor_list, n_step, iter_period)
end

function update_neighbor(Q2d_N::Q2d_Cell_List, x_coords::Vector)
    n_atoms = size(x_coords)[1]

    Q2d_N.n_step += 1
    # x_coords_new = copy(x_coords)
    if (Q2d_N.n_step % Q2d_N.iter_period) == 0
        x_q2d_new = [[x_coords[i][1], x_coords[i][2]] for i in 1:n_atoms]
        update!(Q2d_N.cell_list, x_q2d_new)
        Q2d_N.neighbor_list = neighborlist!(Q2d_N.cell_list)
    end

    return Q2d_N
end