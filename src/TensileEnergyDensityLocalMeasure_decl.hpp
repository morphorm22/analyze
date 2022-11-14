#pragma once

#include "AbstractLocalMeasure.hpp"

namespace Plato
{
/******************************************************************************//**
 * \brief TensileEnergyDensity local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class TensileEnergyDensityLocalMeasure :
        public AbstractLocalMeasure<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using AbstractLocalMeasure<EvaluationType>::mNumSpatialDims;
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms;
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell;
    using AbstractLocalMeasure<EvaluationType>::mSpatialDomain; 

    Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix;

    using StateT   = typename EvaluationType::StateScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mLameConstantLambda, mLameConstantMu, mPoissonsRatio, mYoungsModulus;

    /******************************************************************************//**
     * \brief Get Youngs Modulus and Poisson's Ratio from input parameter list
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void getYoungsModulusAndPoissonsRatio(Teuchos::ParameterList & aInputParams);

    /******************************************************************************//**
     * \brief Compute lame constants for isotropic linear elasticity
    **********************************************************************************/
    void computeLameConstants();

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(
        const Plato::SpatialDomain   & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    );

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aYoungsModulus elastic modulus
     * \param [in] aPoissonsRatio Poisson's ratio
     * \param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(
        const Plato::SpatialDomain & aSpatialModel,
              Plato::DataMap       & aDataMap,
        const Plato::Scalar        & aYoungsModulus,
        const Plato::Scalar        & aPoissonsRatio,
        const std::string          & aName
    );

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~TensileEnergyDensityLocalMeasure();

    /******************************************************************************//**
     * \brief Evaluate tensile energy density local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aDataMap map to stored data
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    void
    operator()(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS
    ) override;
};
// class TensileEnergyDensityLocalMeasure

}
//namespace Plato
