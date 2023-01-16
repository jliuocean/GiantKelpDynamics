using EasyFit, JLD2, Oceananigans, Statistics

function get_observable(path, observed_locations)
    u = FieldTimeSeries(path, "depth_average_u")
    v = FieldTimeSeries(path, "depth_average_v")

    reference_station_u = u[observed_locations[1, 1], observed_locations[2, 1], 1, :]

    reference_station_upstream = reference_station_u .>= 0
    reference_station_downstream = reference_station_u .< 0

    println("$(sum(reference_station_downstream)), $(sum(reference_station_upstream))")

    observable = zeros(3, length(observed_locations[1, :]))

    for station_idx in 1:length(observed_locations[1, :])
        comparison_data_u = u[observed_locations[1, station_idx], observed_locations[2, station_idx], 1, :]
        #comparison_data_v = v[observed_locations[1, station_idx], observed_locations[2, station_idx]]

        std_u = std(comparison_data_u)
        #std_v = std(comparison_data_v)

        if station_idx != 1
            comparison_data_east = comparison_data_u[reference_station_upstream]
            comparison_data_west = comparison_data_u[reference_station_downstream]

            upstream = fitlinear(reference_station_u[reference_station_upstream], comparison_data_east)
            downstream = fitlinear(reference_station_u[reference_station_downstream], comparison_data_west)

            upstream_gradient = upstream.about
            #upstream_r² = upstream.R^2

            downstream_gradient = downstream.about
            #downstream_r² = downstream.R^2
        else
            upstream_gradient, upstream_r², downstream_gradient, downstream_r² = 1.0, 1.0, 1.0, 1.0
        end

        observable[:, station_idx] = [std_u, upstream_gradient, downstream_gradient]
    end

    return observable
end

generation = 1
generation_size = 24

observed_locations = [51 115 115 128 141;
                      102 128 115 128 128]

results_file = jldopen("ensemble_generation_$generation.jld2")

for id in 1:generation_size
    observables = get_observable("calibration_ensemble_$(generation)_$(id).jld2", observed_locations)
    file["$id"] = observables
end

close(results_file)
