#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

class NewmarkIntegrator
{
  protected:
    Plato::Scalar mGamma;
    Plato::Scalar mBeta;
    Plato::Scalar mTimeStep;
    Plato::Scalar mTimeStepScale;
    Plato::Scalar mTerminationTime;
    unsigned int mNumSteps;

  public:
    explicit 
    NewmarkIntegrator(
        const Teuchos::ParameterList & aParams,
              Plato::Scalar            aMaxEigenvalue
    ) :
        mGamma           (aParams.get<double>("Newmark Gamma") ),
        mBeta            (aParams.get<double>("Newmark Beta") ),
        mTimeStep        (0.0),
        mTimeStepScale   (1.0),
        mTerminationTime (0.0),
        mNumSteps        (0)
    {
        bool tTimeStepGiven        = aParams.isType<Plato::Scalar>("Time Step");
        bool tTimeStepScaleGiven   = aParams.isType<Plato::Scalar>("Time Step Scale");
        bool tTerminationTimeGiven = aParams.isType<Plato::Scalar>("Termination Time");
        bool tNumberTimeStepsGiven = aParams.isType<int>("Number Time Steps");

        if ( tNumberTimeStepsGiven && tTerminationTimeGiven )
        {
            throw std::runtime_error("Specify 'Number Time Steps' OR 'Termination Time'. Not both.");
        }

        if ( !tNumberTimeStepsGiven && !tTerminationTimeGiven )
        {
            throw std::runtime_error("Either 'Number Time Steps' or 'Termination Time' must be specified.");
        }

        if ( tNumberTimeStepsGiven )
        {
            mNumSteps = aParams.get<int>("Number Time Steps");
        }
        else
        {
            mTerminationTime = aParams.get<Plato::Scalar>("Termination Time");
        }

        if(this->isConditionallyStable())
        {
            if ( tTimeStepGiven )
            {
                throw std::runtime_error("Time Step cannot be specified for a conditionally stable integrator.");
            }

            Plato::Scalar tCritSamplingFreq = this->getOmega();
            mTimeStep = tCritSamplingFreq / aMaxEigenvalue;

            if ( tTimeStepScaleGiven )
            {
                mTimeStepScale = aParams.get<Plato::Scalar>("Time Step Scale");
            }
            else
            {
                mTimeStepScale = 0.5; // default timestep scale
            }
        }
        else
        {
            if ( tTimeStepGiven )
            {
                mTimeStep = aParams.get<Plato::Scalar>("Time Step");
            }
            else
            {
                throw std::runtime_error("Unconditionally stable integrator requires a 'Time Step' spec.");
            }

            if ( tTimeStepScaleGiven )
            {
                throw std::runtime_error("Unconditionally stable integrator doesn't accept a time step scale.");
            }
        }

        if (mTerminationTime)
        {
            mNumSteps = mTerminationTime / this->getTimeStep() + 1;
        }
        else
        {
            mTerminationTime = mNumSteps * this->getTimeStep();
        }
    }


    ~NewmarkIntegrator() {}

    Plato::Scalar
    getOmega() const
    {
        return pow(mGamma/2.0 - mBeta, -1.0/2.0);
    }

    bool
    isConditionallyStable() const
    {
        return mBeta < mGamma/2.0;
    }

    decltype(mNumSteps)
    getNumSteps() const
    {
        return mNumSteps;
    }

    decltype(mTimeStep)
    getTimeStep() const
    {
        return mTimeStep * mTimeStepScale;
    }

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

    virtual Plato::ScalarVector 
    u_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) = 0;

    virtual Plato::ScalarVector 
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) = 0;

    virtual Plato::ScalarVector 
    a_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) = 0;

};

class NewmarkIntegratorUForm : public NewmarkIntegrator
{
    using NewmarkIntegrator::mGamma;
    using NewmarkIntegrator::mBeta;

  public:
    explicit 
    NewmarkIntegratorUForm(
        const Teuchos::ParameterList & aParams,
              Plato::Scalar            aMaxEigenvalue
    ) :
        NewmarkIntegrator(aParams, aMaxEigenvalue)
    {
    }

    ~NewmarkIntegratorUForm()
    {
    }

    Plato::Scalar v_grad_a ( Plato::Scalar aTimeStep ) override { return 0; }
    Plato::Scalar u_grad_a ( Plato::Scalar aTimeStep ) override { return 0; }

    Plato::Scalar
    v_grad_u( Plato::Scalar aTimeStep ) override
    {
        return -mGamma/(mBeta*aTimeStep);
    }

    Plato::Scalar
    v_grad_u_prev( Plato::Scalar aTimeStep ) override
    {
        return mGamma/(mBeta*aTimeStep);
    }

    Plato::Scalar
    v_grad_v_prev( Plato::Scalar aTimeStep ) override
    {
        return mGamma/mBeta - 1.0;
    }

    Plato::Scalar
    v_grad_a_prev( Plato::Scalar aTimeStep ) override
    {
        return (mGamma/(2.0*mBeta) - 1.0) * aTimeStep;
    }

    Plato::Scalar
    a_grad_u( Plato::Scalar aTimeStep ) override
    {
        return -1.0/(mBeta*aTimeStep*aTimeStep);
    }

    Plato::Scalar
    a_grad_u_prev( Plato::Scalar aTimeStep ) override
    {
        return 1.0/(mBeta*aTimeStep*aTimeStep);
    }

    Plato::Scalar
    a_grad_v_prev( Plato::Scalar aTimeStep ) override
    {
        return 1.0/(mBeta*aTimeStep);
    }

    Plato::Scalar
    a_grad_a_prev( Plato::Scalar aTimeStep ) override
    {
        return 1.0/(2.0*mBeta) - 1.0;
    }

    Plato::ScalarVector 
    u_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override { return Plato::ScalarVector(); }

    Plato::ScalarVector 
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override
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

    Plato::ScalarVector 
    a_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override
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

class NewmarkIntegratorAForm : public NewmarkIntegrator
{
    using NewmarkIntegrator::mGamma;
    using NewmarkIntegrator::mBeta;

  public:
    explicit 
    NewmarkIntegratorAForm(
        const Teuchos::ParameterList & aParams,
              Plato::Scalar            aMaxEigenvalue
    ) :
        NewmarkIntegrator(aParams, aMaxEigenvalue)
    {
    }

    ~NewmarkIntegratorAForm()
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

    Plato::Scalar
    v_grad_a( Plato::Scalar aTimeStep ) override
    {
        return -mGamma*aTimeStep;
    }

    Plato::Scalar
    u_grad_a( Plato::Scalar aTimeStep ) override
    {
        return -mBeta*aTimeStep*aTimeStep;
    }

    Plato::ScalarVector 
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override
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

    Plato::ScalarVector 
    u_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override
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

    Plato::ScalarVector 
    a_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt) override { return Plato::ScalarVector(); }

};

} // namespace Plato
