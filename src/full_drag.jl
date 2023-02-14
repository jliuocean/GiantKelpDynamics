
struct LocalTransform{X, RZ, RX, RN, RP}
    x⃗₀ :: X
    R₁ :: RZ
    R₂ :: RX
    R₃⁺ :: RP
    R₃⁻ :: RN
end

@inline function (transform::LocalTransform)(x, y, z)
    x⃗_ = @inbounds transform.R₁ * (transform.R₂ * ([x, y, z] - transform.x⃗₀))

    if @inbounds x⃗_[3] >= 0.0
        x_, y_, z_ = transform.R₃⁺ * x⃗_
        ± = true
    else
        x_, y_, z_ = transform.R₃⁻ * x⃗_
        ± = false
    end

    return x_, y_, z_, ±
end


@inline function LocalTransform(θ::Number, ϕ::Number, θ⁺::Number, θ⁻::Number, x⃗₀::Vector) 
    R₁ = LinearAlgebra.Givens(1, 3, cos(-ϕ), sin(-ϕ))
    R₂ = LinearAlgebra.Givens(1, 2, cos(θ), sin(θ))

    R₃⁺ = LinearAlgebra.Givens(1, 3, cos(θ⁺), sin(θ⁺))
    R₃⁻ = LinearAlgebra.Givens(1, 3, cos(θ⁻), sin(θ⁻))
    return LocalTransform(x⃗₀, R₁, R₂, R₃⁺, R₃⁻)
end

@kernel function weights!(drag_field, grid, rᵉ, l⁺, l⁻, polar_transform, parameters)
    i, j, k = @index(Global, NTuple)

    x, y, z = @inbounds node(Center(), Center(), Center(), i, j, k, grid)

    if @inbounds ((x - polar_transform.x⃗₀[1]) ^ 2 + (y - polar_transform.x⃗₀[2])^2 + (z - polar_transform.x⃗₀[3])^2) < (max(l⁺, l⁻) + rᵉ) ^ 2
        x_, y_, z_, ± = polar_transform(x, y, z)
        r = (x_ ^ 2 + y_ ^ 2) ^ 0.5
        l = ifelse(±, l⁺, l⁻)
        # who knows why z is signed wrong
        weight = ifelse((r < rᵉ) & (- l < z_), parameters.drag_smoothing(r, rᵉ), 0.0)
        @inbounds drag_field[i, j, k] = weight
    else
        @inbounds drag_field[i, j, k] = 0.0
    end
end

@kernel function apply_drag!(water_accelerations, drag_field, normalisations, grid, Fᴰ, scalefactor, parameters)
    i, j, k = @index(Global, NTuple)

    vol = Vᶜᶜᶜ(i, j, k, grid)
    inverse_effective_mass = @inbounds drag_field[i, j, k] / (normalisations * vol * parameters.ρₒ)
    if any(isnan.(Fᴰ .* inverse_effective_mass)) error("NaN from $Fᴰ, $normalisations * $vol * .../$(drag_nodes[i, j, k])") end
    @inbounds begin
        water_accelerations.u[i, j, k] -= Fᴰ[1] * inverse_effective_mass * scalefactor
        water_accelerations.v[i, j, k] -= Fᴰ[2] * inverse_effective_mass * scalefactor
        water_accelerations.w[i, j, k] -= Fᴰ[3] * inverse_effective_mass * scalefactor
    end
end

@kernel function drag_node!(base_x, base_y, base_z, 
                            scalefactors, positions, positions_ijk,
                            effective_radii, 
                            drag_forces, 
                            grid, drag_field, 
                            n_nodes, 
                            water_accelerations, 
                            weights_kernel!, 
                            apply_drag_kernel!, 
                            parameters)
                            
    p = @index(Global)
    scalefactor = @inbounds scalefactors[p]

    for n = 1:n_nodes
        # get node positions and size
        @inbounds begin
            x⃗ = @inbounds positions[p][n, :] 
            if n==1
                x⃗⁻ =  @inbounds zeros(3)
            else
                x⃗⁻ = @inbounds positions[p][n - 1, :]
            end

            if n == n_nodes
                x⃗⁺ = x⃗
            else
                x⃗⁺ = @inbounds positions[p][n + 1, :]
            end
        end

        rᵉ = @inbounds effective_radii[p][n]

        Δx⃗ = x⃗⁺ - x⃗⁻
                
        if n == 1
            l⁻ = @inbounds sqrt(dot(positions[p][n, :], positions[p][n, :]))
        else
            dp = @inbounds positions[p][n, :] - positions[p][n - 1, :]
            l⁻ = sqrt(dot(dp, dp))/2
        end
            
        θ = @inbounds atan(Δx⃗[2] / (Δx⃗[1] + eps(0.0))) + π * 0 ^ (1 + sign(Δx⃗[1]))
        ϕ = @inbounds atan((Δx⃗[1] ^ 2 + Δx⃗[2] ^ 2 + eps(0.0)) ^ 0.5 / Δx⃗[3])

        
        cosθ⁻ = dot(Δx⃗, x⃗ - x⃗⁻) / (sqrt(dot(Δx⃗, Δx⃗)) * sqrt(dot(x⃗ - x⃗⁻, x⃗ - x⃗⁻)))
        θ⁻ = -1.0 <= cosθ⁻ <= 1.0 ? acos(cosθ⁻) : 0.0

        if n == n_nodes
            dp = @inbounds positions[p][n, :] - positions[p][n - 1, :]
            l⁺ = sqrt(dot(dp, dp))/2
            θ⁺ = θ⁻
        else
            dp = @inbounds positions[p][n + 1, :] - positions[p][n, :]
            l⁺ = sqrt(dot(dp, dp))/2
            cosθ⁺ = - dot(Δx⃗, x⃗⁺ - x⃗) / (sqrt(dot(Δx⃗, Δx⃗)) * sqrt(dot(x⃗⁺ - x⃗, x⃗⁺ - x⃗)))
            θ⁺ = -1.0 <= cosθ⁺ <= 1.0 ? acos(cosθ⁺) : 0.0
        end

        weights_event = @inbounds weights_kernel!(drag_field[p], grid, rᵉ, l⁺, l⁻, LocalTransform(θ, ϕ, θ⁺, θ⁻, x⃗ + [base_x[p], base_y[p], base_z[p]]), parameters)
        wait(weights_event)

        normalisation = @inbounds sum(drag_field[p])

        Fᴰ = @inbounds drag_forces[p][n, :]

        # fallback if nodes are closer together than gridpoints and the line joining them is parallel to a grid Axis
        # as this means there are then no nodes in the stencil. This is mainly an issue for nodes close together lying on the surface
        # As long as the (relaxed) segment lengths are properly considered this shouldn't be an issue except during startup where upstream 
        # elements will quickly move towards dowmnstream elements
        @inbounds if normalisation == 0.0
            @warn "Used fallback drag application as stencil found no nodes, this should be concerning if not in the initial transient response at $p, $n"
            i, j, k = positions_ijk[p][n, :]
            vol = Vᶜᶜᶜ(i, j, k, grid)
            inverse_effective_mass = 1 / (vol * parameters.ρₒ)
            water_accelerations.u[i, j, k] -= Fᴰ[1] * inverse_effective_mass * scalefactor
            water_accelerations.v[i, j, k] -= Fᴰ[2] * inverse_effective_mass * scalefactor
            water_accelerations.w[i, j, k] -= Fᴰ[3] * inverse_effective_mass * scalefactor

        else
            apply_drag_event = @inbounds apply_drag_kernel!(water_accelerations, drag_field[p], normalisation, grid, Fᴰ, scalefactor, parameters)
            wait(apply_drag_event)
        end
    end
end

function fully_resolved_drag!(model)
    particles = model.particles
    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    workgroup, worksize = work_layout(model.grid, :xyz)
    weights_kernel! = weights!(device(model.architecture), workgroup, worksize)
    apply_drag_kernel! = apply_drag!(device(model.architecture), workgroup, worksize)

    n_particles = length(particles)
    n_nodes = particles.parameters.n_nodes

    drag_water_kernel! = drag_node!(device(model.architecture), min(256, n_particles), n_particles)

    drag_nodes_event = drag_water_kernel!(particles.properties.x, particles.properties.y, particles.properties.z, 
                                          particles.properties.scalefactor, 
                                          particles.properties.positions, 
                                          particles.properties.effective_radii, 
                                          particles.properties.drag_forces, 
                                          model.grid, 
                                          particles.properties.drag_field, 
                                          n_nodes, 
                                          water_accelerations, 
                                          weights_kernel!, 
                                          apply_drag_kernel!, 
                                          particles.parameters)
    wait(drag_nodes_event)
end