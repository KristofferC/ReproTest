module ReproTest

import LinearAlgebra: LinearAlgebra, BLAS, BLAS.@blasfunc, BlasInt
const liblapack = LinearAlgebra.LAPACK.liblapack
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

for (larfg, elty) in
    ((:dlarfg_, Float64),
     (:slarfg_, Float32),
     (:zlarfg_, ComplexF64),
     (:clarfg_, ComplexF32))
    @eval begin
        #        .. Scalar Arguments ..
        #        INTEGER            incx, n
        #        DOUBLE PRECISION   alpha, tau
        #        ..
        #        .. Array Arguments ..
        #        DOUBLE PRECISION   x( * )
        function larfg!(x::AbstractVector{$elty})
            N    = BlasInt(length(x))
            α    = Ref{$elty}(x[1])
            incx = BlasInt(1)
            τ    = Ref{$elty}(0)
            ccall((@blasfunc($larfg), liblapack), Cvoid,
                (Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}),
                N, α, pointer(x, 2), incx, τ)
            @inbounds x[1] = one($elty)
            return τ[]
        end
    end
end

@show ReproTest.larfg!(rand(100))


end # module ReproTest
