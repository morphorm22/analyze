#pragma once

#include <memory>
#include <algorithm>

#include "base/CriterionBase.hpp"
#include "elliptic/AbstractLocalMeasure.hpp"


namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Volume integral criterion of field quantites (primarily for use with VolumeAverageCriterion)
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class CriterionVolumeIntegral : public Plato::CriterionBase
{
private:
  using ElementType = typename EvaluationType::ElementType;  
  
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;  
  
  using CriterionBaseType = Plato::CriterionBase;
  using CriterionBaseType::mSpatialDomain; /*!< mesh database */
  using CriterionBaseType::mDataMap; /*!< PLATO Engine output database */  

  using StateT   = typename EvaluationType::StateScalarType;   /*!< state variables automatic differentiation type */
  using ConfigT  = typename EvaluationType::ConfigScalarType;  /*!< configuration automatic differentiation type */
  using ResultT  = typename EvaluationType::ResultScalarType;  /*!< result variables automatic differentiation type */
  using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */

   std::string mSpatialWeightFunction;
   Plato::Scalar mPenalty;        /*!< penalty parameter in SIMP model */
   Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */
   std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>> mLocalMeasure; /*!< Volume averaged quantity with evaluation type */

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams);

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams);

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aPlatoDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName user defined function name
     **********************************************************************************/
    CriterionVolumeIntegral(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    );

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    CriterionVolumeIntegral(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    );

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~CriterionVolumeIntegral();

    /******************************************************************************//**
     * \brief Set volume integrated quanitity
     * \param [in] aInputEvaluationType evaluation type volume integrated quanitity
    **********************************************************************************/
    void setVolumeIntegratedQuantity(const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInput);

    /******************************************************************************//**
     * \brief Set spatial weight function
     * \param [in] aInput math expression
    **********************************************************************************/
    void setSpatialWeightFunction(std::string aWeightFunctionString) override;

    /// @fn isLinear
    /// @brief returns true if criterion is linear
    /// @return boolean
    bool 
    isLinear() 
    const;

    /// @fn updateProblem
    /// @brief update criterion parameters at runtime
    /// @param [in] aWorkSets function domain and range workset database
    /// @param [in] aCycle    scalar 
    virtual 
    void 
    updateProblem(
      const Plato::WorkSets & aWorkSets,
      const Plato::Scalar   & aCycle
    ) override;

    /// @fn evaluateConditional
    /// @brief Evaluate volume average criterion
    /// @param [in] aWorkSets function domain and range workset database
    /// @param [in] aCycle    scalar 
    void evaluateConditional(
      const Plato::WorkSets & aWorkSets,
      const Plato::Scalar   & aCycle
    ) const;
};
// class CriterionVolumeIntegral

}
//namespace Elliptic

}
//namespace Plato
