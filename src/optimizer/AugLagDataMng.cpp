/*
 * AugLagDataMng.cpp
 *
 *  Created on: May 4, 2023
 */

#include "BLAS1.hpp"

#include "AugLagDataMng.hpp"

namespace Plato
{
    void AugLagDataMng::initialize()
    {
        Plato::blas1::fill(mInitiaPenalty, mPenaltyValues);
        Plato::blas1::fill(mInitialLagrangeMultiplier, mLagrangeMultipliers);
    }

    void AugLagDataMng::updateLagrangeMultipliers()
    {
        // copy global member data to local scope
        Plato::ScalarVector tPenaltyValues = mPenaltyValues;
        Plato::ScalarVector tLagrangeMultipliers  = mLagrangeMultipliers;
        Plato::ScalarVector tCurrentConstraintValues = mCurrentConstraintValues;
        // update lagrange multipliers
        const Plato::OrdinalType tNumCells = mNumCells;
        Plato::OrdinalType tNumLocalConstraints = mNumLocalConstraints;
        Kokkos::parallel_for("update lagrange multipliers",Kokkos::RangePolicy<>(0,tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
            {
                Plato::OrdinalType tLocalIndex = (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;
                tLagrangeMultipliers(tLocalIndex) = tLagrangeMultipliers(tLocalIndex) 
                    + ( tPenaltyValues(tLocalIndex) * tCurrentConstraintValues(tLocalIndex) );
            }
        });
    }

    void AugLagDataMng::updatePenaltyValues()
    {
        // copy member containers to local scope
        Plato::ScalarVector tPenaltyValues  = mPenaltyValues;
        Plato::ScalarVector tCurrentConstraintValues = mCurrentConstraintValues;
        Plato::ScalarVector tPreviousConstraintValues = mPreviousConstraintValues;
        // update penalty values
        const Plato::OrdinalType tNumCells = mNumCells;
        Plato::OrdinalType tNumLocalConstraints = mNumLocalConstraints;
        Kokkos::parallel_for("update penalty values",Kokkos::RangePolicy<>(0,tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
            {
                Plato::OrdinalType tLocalIndex = (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;   
                // evaluate condition
                Plato::Scalar tCondition = mPenaltyUpdateParameter * tPreviousConstraintValues(tLocalIndex);
                Plato::Scalar tTrialPenalty = tCurrentConstraintValues(tLocalIndex) > tCondition ? 
                    mPenaltyIncrement * tPenaltyValues(tLocalIndex) : tPenaltyValues(tLocalIndex); 
                tPenaltyValues(tLocalIndex) = tTrialPenalty < mMaxPenalty ? tTrialPenalty : mMaxPenalty;
            }
        });
    }

    void AugLagDataMng::parseNumerics(Teuchos::ParameterList &aParams)
    {
        mMaxPenalty = aParams.get<Plato::Scalar>("Maximum Penalty", 10000.0);
        mInitiaPenalty = aParams.get<Plato::Scalar>("Initial Penalty", 1.0);
        mPenaltyIncrement = aParams.get<Plato::Scalar>("Penalty Increment", 1.1);
        mPenaltyUpdateParameter = aParams.get<Plato::Scalar>("Penalty Update Constant", 0.25);
        mInitialLagrangeMultiplier = aParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.0);
    }

    void AugLagDataMng::parseLimits(Teuchos::ParameterList &aParams)
    {
        bool tIsArray = aParams.isType<Teuchos::Array<Plato::Scalar>>("Limits");
        if (tIsArray)
        {
            // parse inputs
            Teuchos::Array<Plato::Scalar> tLimits = aParams.get<Teuchos::Array<Plato::Scalar>>("Limits");
            // create mirror 
            mNumLocalConstraints = tLimits.size();
            mLocalMeasureLimits  = Plato::ScalarVector("Limits",mNumLocalConstraints);
            auto tHostArray = Kokkos::create_mirror_view(mLocalMeasureLimits);
            for( Plato::OrdinalType tIndex = 0; tIndex < mNumLocalConstraints; tIndex++)
            {
                tHostArray(tIndex) = tLimits[tIndex];
            }
            Kokkos::deep_copy(mLocalMeasureLimits,tHostArray);
        }
        else
        {
            auto tMsg = std::string("Local constraints limits are not defined in criterion block. ") 
                + "Constraint limits are required to properly enforce the local constraints.";
            ANALYZE_THROWERR(tMsg)
        }
    }

    void AugLagDataMng::allocateContainers(const Plato::OrdinalType &aNumCells)
    {
        if(mNumLocalConstraints <= 0)
        {
            auto tMsg = std::string("Number of local constraints is not defined; i.e., ") 
                + "number of local constraints is set to " + std::to_string(mNumLocalConstraints) + ".";
            ANALYZE_THROWERR(tMsg)
        }
        mNumCells = aNumCells;
        if(mNumCells <= 0)
        {
            auto tMsg = std::string("Number of cells is not defined; i.e., ") 
                + "number of cells is set to " + std::to_string(mNumCells) + ".";
            ANALYZE_THROWERR(tMsg)
        }

        auto tNumConstraints = mNumLocalConstraints * mNumCells;
        mPenaltyValues = Plato::ScalarVector("Penalty Values",tNumConstraints);
        mLagrangeMultipliers = Plato::ScalarVector("Lagrange Multipliers",tNumConstraints);
        mCurrentConstraintValues = Plato::ScalarVector("Current Constraints",tNumConstraints);
        mPreviousConstraintValues = Plato::ScalarVector("Previous Constraints",tNumConstraints);
    }
}
// namespace Plato