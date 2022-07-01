#pragma once

#include "ThermoelasticMaterial.hpp"
#include "AbstractLocalMeasure.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief VonMises local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class ThermalVonMisesLocalMeasure :
        public AbstractLocalMeasure<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using AbstractLocalMeasure<EvaluationType>::mNumSpatialDims;
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms;
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell;
    using AbstractLocalMeasure<EvaluationType>::mSpatialDomain; 

    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

    static constexpr Plato::OrdinalType TDofOffset = mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumDofsPerNode = ElementType::mNumDofsPerNode;

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    ThermalVonMisesLocalMeasure(
        const Plato::SpatialDomain   & aSpatialDomain,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    );

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~ThermalVonMisesLocalMeasure();

    /******************************************************************************//**
     * \brief Evaluate vonmises local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aDataMap map to stored data
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    virtual void
    operator()(
        const Plato::ScalarMultiVectorT <StateT>  & aStateWS,
        const Plato::ScalarArray3DT     <ConfigT> & aConfigWS,
              Plato::ScalarVectorT      <ResultT> & aResultWS
    ) override;
};
// class ThermalVonMisesLocalMeasure

}
//namespace Plato
