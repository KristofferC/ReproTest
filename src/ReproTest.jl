module ReproTest

const liblapack = Base.liblapack_name
import LinearAlgebra: BLAS.@blasfunc, BlasInt

## Tools to compute and apply elementary reflectors
for (larfg, elty) in
    ((:dlarfg_, Float64),
     (:slarfg_, Float32),
     (:zlarfg_, ComplexF64),
     (:clarfg_, ComplexF32))
    @eval begin
        #
        #    larfg!(x) -> (τ, β)
        #
        # Wrapper to LAPACK function family _LARFG.F to compute the parameters
        # v and τ of a Householder reflector H = I - τ*v*v' which annihilates
        # the N-1 trailing elements of x and to provide τ and β, where β is the
        # first element of the transformed vector H*x. The vector v has its first
        # component set to 1 and is returned in x.
        #
        #        .. Scalar Arguments ..
        #        INTEGER            incx, n
        #        DOUBLE PRECISION   alpha, tau
        #        ..
        #        .. Array Arguments ..
        #        DOUBLE PRECISION   x( * )
        function larfg!(x::AbstractVector{$elty})
            N    = BlasInt(length(x))
            incx = stride(x, 1)
            τ    = Ref{$elty}(0)
            ccall((@blasfunc($larfg), liblapack), Cvoid,
                (Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}),
                N, x, pointer(x, 2), incx, τ)
            β = x[1]
            @inbounds x[1] = one($elty)
            return τ[], β
        end
    end
end

ReproTest.larfg!(rand(100))


end # module ReproTest
