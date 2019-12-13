using LazIO
using LasIO
using FileIO
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using DataFrames
using PointCloudRasterizers


"Store coordinates"
function coordinates(points,h)
    coords = SArray{Tuple{9},Float64,1,9}[]
    for (ind,p) in enumerate(points)
        x = xcoord(p,h)
        y = ycoord(p,h)
        z = zcoord(p,h)
        inten = intensity(p)
        ret = return_number(p)
        nret = number_of_returns(p)
        classificc = classification(p)
        gps = gps_time(p)
        push!(coords,(ind, x, y, z, inten, ret, nret, classificc, gps))
    end
    return coords
end

"Convert Cartesian to Spherical coordinates"
function cart2pol(x::Float64, y::Float64, z::Float64)
    cart = Tuple{Float64,Float64,Float64}
    ρ = sqrt(x^2 + y^2 + z^2) #distance
    θ = atand(y,x) #horizontal angle
    ϕ = acosd(z/ρ) #incidence angle
    cart = (ρ,θ,ϕ)
    return cart
end

"Convert Spherical to Cartesian coordinates"
function pol2cart(ρ::Float64, θ::Float64, ϕ::Float64)
    pol = Tuple{Float64,Float64,Float64}
    x = (ρ * sind(ϕ) * cosd(θ))
    y = (ρ * sind(ϕ) * sind(θ))
    z = (ρ * cosd(ϕ))
    return pol = (x,y,z)
end

const c_air, c_water = 2.99*10^8, 2.25*10^8
const n_air_water = c_water/c_air

"Correct an underwater point according to a watersurface point from the refraction factor by using Snell's Law."
function refraction_correction(watersurface::SArray{Tuple{4},Float64,1,4},underwater::SArray{Tuple{4},Float64,1,4})
    x0, y0, z0 = watersurface[2:4]
    x1, y1, z1 = underwater[2:4]
    Δx, Δy, Δz = (x1-x0), (y1-y0), (z1-z0)
    ind = underwater[1]
    #Calculate the distance of the two points (ρ), the vertical-incidence angle (ϕ) and horizontal angle (θ) between the two points
    ρ,θ,ϕ = cart2pol(Δx, Δy, Δz)

    #Calculate the underwater angle due to the refraction effect.
    #Using Snell's Law: n(air)*sin(ϕ_air) = n(water)*sin(ϕ_water),
    #where ϕ_air: the angle of incidence (I know that the angle of the scan is 20)
    #      ϕ_water: the angle of refraction
    #      n(air): the refraction idex of medium containing the incident ray, value=1.000293
    #      n(water): the refraction idex of medium containing the transmitted ray, value=1.333

    #check the height values of the water-surface and underwater point
    if z0 == z1
        error("the water-surface and underwater point are at the same height level")
    elseif z0 > z1
        ϕ_water = asind(n_air_water*sind(ϕ)) #ϕ_new is the new vertical
    else
        error("the underwater point can not be higher than the watersurface point")
    end

    #Distance under the water has been influenced by the different speed of light in the water.
    #In a specific moment, ρ = c_air * t and ρ_water = c_water * t.
    #So, by diving these two equations, the ratio (ρ1/ρ2) = n_air_water
    ρ_water = (ρ*n_air_water)

    #Calculate the correction in the 3d space using the pol2cart function
    xcor, ycor, zcor = pol2cart(ρ_water, θ, ϕ_water)

    #Define the new corrected coordinates of the points
    xnew = x0 + xcor
    ynew = y0 + ycor
    znew = z0 - zcor

    #Return the new corrected point
    corrections = @SVector[xcor, ycor, zcor]
    correct_point = @SVector[ind, xnew, ynew ,znew]

    return corrections, correct_point
end

"Group the laser pulses based on their GPS time"
function laserpulses(points)
    println("Execution Time of laser pulses " * string(length(points)) * " points")
    @time begin
        laser_pulses = SArray{Tuple{}}[] #specify the type of array
        sorted_gps = sort!(points, by = x -> x[9], rev = false)
        pp = [tuple(p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]) for p in sorted_gps]
        df = DataFrame(pp[:]) #create a dataframe
        groups = groupby(df,:9) #group by gps time
        for gr in groups
            if size(gr,1) > 1
                push!(laser_pulses,[gr])
            end
        end
    end
    return laser_pulses
end

"Calculate the corrected/refracted points for every pulse"
function corrected_points(laser_pulses,coords)
    corrected_laser_pulses = [] #specify the type of array

    INDEXES_corr = [] #indexes of corrected_points
    INDEXES_un_corr = [] #indexes of non-corrected points

    for every_pulse in laser_pulses
        indexes = getindex(every_pulse)[1]
        ind_surf = Int64(indexes[1]) #water surface index
        p_surf = coords[ind_surf] #water surface point
        p_surf = @SVector[ind_surf,p_surf[2],p_surf[3],p_surf[4]] #create Arrays
        corrected_points = SArray{Tuple{4},Float64,1,4}[] #store the corrected points for each pulse!
        not_corrected = SArray{Tuple{4},Float64,1,4}[] #store the un-corrected points due to the Exception ERRORsfor each pulse!

        push!(corrected_points,(p_surf)) # add the water surface point
        push!(not_corrected,(p_surf)) # add the water surface point

        for ind_under = 2:length(indexes) #iterate throught the inder water points
            n = Int64(indexes[ind_under])
            p_under = coords[n]
            p_under = @SVector[n,p_under[2],p_under[3],p_under[4]]
            try
                correction, corre_pt = refraction_correction(p_surf,p_under)
                push!(INDEXES_corr,p_under[1])
                push!(corrected_points,(corre_pt)) # I will classify as un-refracted with classif.code = 19
            catch
                push!(INDEXES_un_corr,p_under[1])
                push!(not_corrected,(p_under)) # I will classify as un-refracted with classif.code = 20
            end
        end

        if length(corrected_points) > 1 #check if there is NOT only water surface point
            push!(corrected_laser_pulses,[corrected_points])
        end
    end
    return corrected_laser_pulses, INDEXES_corr, INDEXES_un_corr
end


#Write a new classified Pointcloud
function write_pointcloud(ds,header,pp,indexes_corr,index_un_corr)
    n = length(indexes_corr) + length(index_un_corr)
    println("Execution Time of writing points " * string(n) * " points")
    @time  begin
        laz_out = joinpath(path, "myoutput_" * string(length(pp)) * "_points.laz")
        LazIO.write(laz_out, ds.header) do io
            for (ind,p) in enumerate(ds)
                if ind in indexes_corr
                    p.classification = UInt8(19)
                elseif ind in indexes_un_corr #this elseif doesn't work!!why???
                    p.classification = UInt8(20)
                else
                    p.classification = UInt8(9)
                end
                LazIO.writepoint(io, p)
            end
        end
    end
end


#MAIN#
const path = "C:/Users/alexandr/OneDrive - Stichting Deltares/Desktop/NL3_subset/"
const filename = "NL3_clipped_water_cropheight_pointformat3_classification.laz" #classification = 9 (water)
const lazinput = path * filename

#Open LAZ pointcloud
open_file = File{format"LAZ_"}(lazinput)
h, points = load(open_file)
n = length(points)

#dataset
ds = LazIO.open(lazinput)

#subset
pp = points[1500000:1520000]
#Array with coordinates
coords = coordinates(pp,h)
#Separate laser pulses
laser_pulses = laserpulses(coords)
#Return corrected points
corrected_laser_pulses, indexes_corr, indexes_un_corr  = corrected_points(laser_pulses,coords)
#write pointcloud
write_pointcloud(ds,h,pp,indexes_corr,indexes_un_corr)
