#pragma once

#include <memory>

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include <Teuchos_ParameterList.hpp>
#include "elliptic/evaluators/criterion/CriterionEvaluatorBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Scalar function base factory
 **********************************************************************************/
template<typename PhysicsType>
class FactoryCriterionEvaluator
{
public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    FactoryCriterionEvaluator () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~FactoryCriterionEvaluator() {}

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase> 
    create(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aFunctionName);
};
// class FactoryCriterionEvaluator

} // namespace Elliptic

} // namespace Plato
