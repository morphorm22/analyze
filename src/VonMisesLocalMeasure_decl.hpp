#pragma once

#include "PlatoStaticsTypes.hpp"
#include "ElasticModelFactory.hpp"
#include "AbstractLocalMeasure.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief VonMises local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class VonMisesLocalMeasure :
    public AbstractLocalMeasure<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using AbstractLocalMeasure<EvaluationType>::mNumSpatialDims;
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms;
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell;
    using AbstractLocalMeasure<EvaluationType>::mSpatialDomain; 

    using MatrixType = Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms>;
    MatrixType mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT   = typename EvaluationType::StateScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;

    Plato::DataMap mDataMap;

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    );

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aCellStiffMatrix stiffness matrix
     * \param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap,
        const MatrixType           & aCellStiffMatrix,
        const std::string            aName
    );

    /******************************************************************************//**
     * \brief Evaluate vonmises local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    void operator()(
        const Plato::ScalarMultiVectorT<StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT<ControlT> & aControlWS,
        const Plato::ScalarArray3DT<ConfigT>      & aConfigWS,
              Plato::ScalarVectorT<ResultT>       & aResultWS
    ) override;
};
// class VonMisesLocalMeasure

}
//namespace Plato
