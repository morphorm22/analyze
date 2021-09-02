#ifndef NEWMARK_HPP
#define NEWMARK_HPP

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class NewmarkIntegrator
/******************************************************************************/
{
  protected:
    Plato::Scalar mGamma;
    Plato::Scalar mBeta;

  public:
    /******************************************************************************/
    explicit 
    NewmarkIntegrator(Teuchos::ParameterList& aParams) :
      mGamma( aParams.get<double>("Newmark Gamma") ),
      mBeta ( aParams.get<double>("Newmark Beta") )
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    ~NewmarkIntegrator() {}
    /******************************************************************************/

    virtual Plato::Scalar v_grad_a      ( Plato::Scalar aTimeStep ) = 0;
    virtual Plato::Scalar u_grad_a      ( Plato::Scalar aTimeStep ) = 0;

    virtual Plato::Scalar v_grad_u      ( Plato::Scalar aTimeStep ) = 0;
    virtual Plato::Scalar v_grad_u_prev ( Plato::Scalar aTimeStep ) = 0;
    virtual Plato::Scalar v_grad_v_prev ( Plato::Scalar aTimeStep ) = 0;
    virtual Plato::Scalar v_grad_a_prev ( Plato::Scalar aTimeStep ) = 0;
    virtual Plato::Scalar a_grad_u      ( Plato::Scalar aTimeStep ) = 0;
    virtual Plato::Scalar a_grad_u_prev ( Plato::Scalar aTimeStep ) = 0;
    virtual Plato::Scalar a_grad_v_prev ( Plato::Scalar aTimeStep ) = 0;
    virtual Plato::Scalar a_grad_a_prev ( Plato::Scalar aTimeStep ) = 0;

    /******************************************************************************/
    virtual Plato::ScalarVector 
    u_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) = 0;

    /******************************************************************************/
    virtual Plato::ScalarVector 
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) = 0;

    /******************************************************************************/
    virtual Plato::ScalarVector 
    a_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) = 0;

};

/******************************************************************************/
template<typename EvaluationType>
class NewmarkIntegratorUForm : public NewmarkIntegrator<EvaluationType>
/******************************************************************************/
{
    using NewmarkIntegrator<EvaluationType>::mGamma;
    using NewmarkIntegrator<EvaluationType>::mBeta;

  public:
    /******************************************************************************/
    explicit 
    NewmarkIntegratorUForm(
        Teuchos::ParameterList& aParams
    ) :
        NewmarkIntegrator<EvaluationType>(aParams)
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    ~NewmarkIntegratorUForm()
    /******************************************************************************/
    {
    }

    Plato::Scalar v_grad_a ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar u_grad_a ( Plato::Scalar aTimeStep ) override { return 0; }

    /******************************************************************************/
    Plato::Scalar
    v_grad_u( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return -mGamma/(mBeta*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_u_prev( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return mGamma/(mBeta*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_v_prev( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return mGamma/mBeta - 1.0;
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_a_prev( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return (mGamma/(2.0*mBeta) - 1.0) * aTimeStep;
    }

    /******************************************************************************/
    Plato::Scalar
    a_grad_u( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return -1.0/(mBeta*aTimeStep*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    a_grad_u_prev( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return 1.0/(mBeta*aTimeStep*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    a_grad_v_prev( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return 1.0/(mBeta*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    a_grad_a_prev( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return 1.0/(2.0*mBeta) - 1.0;
    }

    /******************************************************************************/
    Plato::ScalarVector 
    u_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override {}


    /******************************************************************************/
    Plato::ScalarVector 
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override
    /******************************************************************************/
    {
        auto tNumData = aU.extent(0);
        Plato::ScalarVector tReturnValue("velocity residual", tNumData);

        auto tGamma = mGamma;
        auto tBeta = mBeta;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumData), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            Plato::Scalar tPredV = aV_prev(aOrdinal) + (1.0-tGamma)*dt*aA_prev(aOrdinal);
            Plato::Scalar tPredU = aU_prev(aOrdinal) + dt*aV_prev(aOrdinal) + dt*dt/2.0*(1.0-2.0*tBeta)* aA_prev(aOrdinal);
            tReturnValue(aOrdinal) = aV(aOrdinal) - tPredV - tGamma/(tBeta*dt)*(aU(aOrdinal) - tPredU);
        }, "Velocity residual value");

        return tReturnValue;
    }

    /******************************************************************************/
    Plato::ScalarVector 
    a_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override
    /******************************************************************************/
    {
        auto tNumData = aU.extent(0);
        Plato::ScalarVector tReturnValue("velocity residual", tNumData);

        auto tBeta = mBeta;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumData), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            Plato::Scalar tPredU = aU_prev(aOrdinal) + dt*aV_prev(aOrdinal) + dt*dt/2.0*(1.0-2.0*tBeta)* aA_prev(aOrdinal);
            tReturnValue(aOrdinal) = aA(aOrdinal) - 1.0/(tBeta*dt*dt)*(aU(aOrdinal) - tPredU);
        }, "Velocity residual value");

        return tReturnValue;
    }
};

/******************************************************************************/
template<typename EvaluationType>
class NewmarkIntegratorAForm : public NewmarkIntegrator<EvaluationType>
/******************************************************************************/
{
    using NewmarkIntegrator<EvaluationType>::mGamma;
    using NewmarkIntegrator<EvaluationType>::mBeta;

  public:
    /******************************************************************************/
    explicit 
    NewmarkIntegratorAForm(
        Teuchos::ParameterList& aParams
    ) :
        NewmarkIntegrator<EvaluationType>(aParams)
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    ~NewmarkIntegratorAForm()
    /******************************************************************************/
    {
    }

    Plato::Scalar v_grad_u      ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar v_grad_u_prev ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar v_grad_v_prev ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar v_grad_a_prev ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar a_grad_u      ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar a_grad_u_prev ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar a_grad_v_prev ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar a_grad_a_prev ( Plato::Scalar aTimeStep ) override { return 0; }


    /******************************************************************************/
    Plato::Scalar
    v_grad_a( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return -mGamma*aTimeStep;
    }

    /******************************************************************************/
    Plato::Scalar
    u_grad_a( Plato::Scalar aTimeStep ) override
    /******************************************************************************/
    {
        return -mBeta*aTimeStep*aTimeStep;
    }

    /******************************************************************************/
    Plato::ScalarVector 
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override
    /******************************************************************************/
    {
        auto tNumData = aU.extent(0);
        Plato::ScalarVector tReturnValue("velocity residual", tNumData);

        auto tGamma = mGamma;
        auto tBeta = mBeta;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumData), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            Plato::Scalar tPredV = aV_prev(aOrdinal) + (1.0-tGamma)*dt*aA_prev(aOrdinal);
            Plato::Scalar tPredU = aU_prev(aOrdinal) + dt*aV_prev(aOrdinal) + dt*dt/2.0*(1.0-2.0*tBeta)* aA_prev(aOrdinal);
            tReturnValue(aOrdinal) = aV(aOrdinal) - tPredV - tGamma*dt*aA(aOrdinal);
        }, "Velocity residual value");

        return tReturnValue;
    }

    /******************************************************************************/
    Plato::ScalarVector 
    u_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override
    /******************************************************************************/
    {
        auto tNumData = aU.extent(0);
        Plato::ScalarVector tReturnValue("velocity residual", tNumData);

        auto tBeta = mBeta;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumData), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            Plato::Scalar tPredU = aU_prev(aOrdinal) + dt*aV_prev(aOrdinal) + dt*dt/2.0*(1.0-2.0*tBeta)* aA_prev(aOrdinal);
            tReturnValue(aOrdinal) = aU(aOrdinal) - tPredU - tBeta*dt*dt*aA(aOrdinal);
        }, "Displacement residual value");

        return tReturnValue;
    }

    /******************************************************************************/
    Plato::ScalarVector 
    a_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override {}

};

} // namespace Plato

#endif
