#pragma once

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * \brief Abstract local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AbstractLocalMeasure :
    public EvaluationType::ElementType
{
protected:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using StateT   = typename EvaluationType::StateScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;

    const Plato::SpatialDomain & mSpatialDomain;
          Plato::DataMap       & mDataMap;

    const std::string mName; /*!< Local measure name */

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    AbstractLocalMeasure(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) :
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap),
        mName          (aName)
    {
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aName local measure name
     **********************************************************************************/
    AbstractLocalMeasure(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap,
        const std::string          & aName
    ) : 
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap),
        mName          (aName)
    {
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~AbstractLocalMeasure()
    {
    }

    /******************************************************************************//**
     * \brief Evaluate local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
    **********************************************************************************/
    virtual void
    operator()(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS ) = 0;

    /******************************************************************************//**
     * \brief Get local measure name
     * \return Return local measure name
     **********************************************************************************/
    virtual std::string getName()
    {
        return mName;
    }
};
//class AbstractLocalMeasure

}
//namespace Plato
