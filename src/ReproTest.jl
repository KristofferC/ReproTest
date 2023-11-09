module ReproTest

const liblapack = Base.liblapack_name
import LinearAlgebra: LinearAlgebra, BLAS, BLAS.@blasfunc, BlasInt
using Libdl

@show @blasfunc(dlarfg_)
@show BLAS.vendor()
@show BLAS.get_num_threads()
@show config = LinearAlgebra.BLAS.lbt_get_config()
for lib in config.loaded_libs
    display(lib)
end
@show Libdl.dlpath(liblapack)
lapack_ptr = Libdl.dlopen(liblapack)
@show Libdl.dlsym(lapack_ptr, @blasfunc(dlarfg_))
@show get(ENV, "LD_LIBRARY_PATH", nothing)



function larfg!(x::AbstractVector{Float64})
    N    = BlasInt(length(x))
    incx = stride(x, 1)
    τ    = Ref{Float64}(0)
    ccall((@blasfunc(dlarfg_), liblapack), Cvoid,
        (Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ref{Float64}),
        N, x, pointer(x, 2), incx, τ)
    β = x[1]
    @inbounds x[1] = one(Float64)
    return τ[], β
end


@show ReproTest.larfg!(rand(100))


end # module ReproTest
