export bulket_sorting

# Notice: in this function we will not consider case where z_i == z_j, which will make the code much more complex
function bulket_sorting(coords::T, L::Tuple) where{T}
    z_coord = [coord[3] for coord in coords]
    n_atoms = size(z_coord)[1]
    L_x, L_y, L_z = L
    dL = L / n_atoms
    
end