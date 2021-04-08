/*
 * AbstractVectorFunction.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"
#include "ExpInstMacros.hpp"

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class AbstractVectorFunction
 *
 * \brief Pure virtual base class for vector functions.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class AbstractVectorFunction
{
private:
    using ResultT = typename EvaluationT::ResultScalarType;

public:
    AbstractVectorFunction(){}
    virtual ~AbstractVectorFunction(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate vector function within the domain.
     *
     * \param [in] aWorkSets holds state and control worksets
     * \param [in/out] aResult   result workset
     ******************************************************************************/
    virtual void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate vector function on non-prescribed boundaries.
     *
     * \param [in] aWorkSets holds state and control worksets
     * \param [in/out] aResult   result workset
     ******************************************************************************/
    virtual void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;

    /***************************************************************************//**
     * \fn void evaluatePrescribed
     * \brief Evaluate vector function on prescribed boundaries.
     *
     * \param [in] aWorkSets holds state and control worksets
     * \param [in/out] aResult   result workset
     ******************************************************************************/
    virtual void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;
};
// class AbstractVectorFunction

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::AbstractVectorFunction, Plato::MassConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::AbstractVectorFunction, Plato::MassConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::AbstractVectorFunction, Plato::MassConservation, Plato::SimplexFluids, 3, 1)
#endif
