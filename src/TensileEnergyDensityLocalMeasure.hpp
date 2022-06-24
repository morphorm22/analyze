#pragma once

#include "AbstractLocalMeasure.hpp"
#include "PlatoMathTypes.hpp"
#include "LinearStress.hpp"
#include "GradientMatrix.hpp"
#include "TensileEnergyDensity.hpp"
#include "SmallStrain.hpp"
#include "ImplicitFunctors.hpp"
#include <Teuchos_ParameterList.hpp>
#include "Eigenvalues.hpp"
#include "BaseExpInstMacros.hpp"

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

    using AbstractLocalMeasure<EvaluationType>::mNumSpatialDims; /*!< space dimension */
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms; /*!< number of voigt tensor terms */
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell; /*!< number of nodes per cell */
    using AbstractLocalMeasure<EvaluationType>::mSpatialDomain; 

    Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    Plato::Scalar mLameConstantLambda, mLameConstantMu, mPoissonsRatio, mYoungsModulus;

    /******************************************************************************//**
     * \brief Get Youngs Modulus and Poisson's Ratio from input parameter list
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void getYoungsModulusAndPoissonsRatio(Teuchos::ParameterList & aInputParams)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();

        auto tModelParamLists = aInputParams.get<Teuchos::ParameterList>("Material Models");
        auto tModelParamList  = tModelParamLists.get<Teuchos::ParameterList>(tMaterialName);

        if( tModelParamList.isSublist("Isotropic Linear Elastic") ){
            Teuchos::ParameterList tParamList = tModelParamList.sublist("Isotropic Linear Elastic");
            mPoissonsRatio = tParamList.get<Plato::Scalar>("Poissons Ratio");
            mYoungsModulus = tParamList.get<Plato::Scalar>("Youngs Modulus");
        }
        else
        {
            throw std::runtime_error("Tensile Energy Density requires Isotropic Linear Elastic Material Model in ParameterList");
        }
    }

    /******************************************************************************//**
     * \brief Compute lame constants for isotropic linear elasticity
    **********************************************************************************/
    void computeLameConstants()
    {
        mLameConstantMu     = mYoungsModulus / 
                             (static_cast<Plato::Scalar>(2.0) * (static_cast<Plato::Scalar>(1.0) + 
                              mPoissonsRatio));
        mLameConstantLambda = static_cast<Plato::Scalar>(2.0) * mLameConstantMu * mPoissonsRatio / 
                             (static_cast<Plato::Scalar>(1.0) - static_cast<Plato::Scalar>(2.0) * mPoissonsRatio);
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(
        const Plato::SpatialDomain   & aSpatialModel,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) : 
        AbstractLocalMeasure<EvaluationType>(aSpatialModel, aInputParams, aName)
    {
        getYoungsModulusAndPoissonsRatio(aInputParams);
        computeLameConstants();
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aYoungsModulus elastic modulus
     * \param [in] aPoissonsRatio Poisson's ratio
     * \param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(
        const Plato::SpatialDomain & aSpatialModel,
        const Plato::Scalar        & aYoungsModulus,
        const Plato::Scalar        & aPoissonsRatio,
        const std::string          & aName
    ) :
        AbstractLocalMeasure<EvaluationType>(aSpatialModel, aName),
        mYoungsModulus(aYoungsModulus),
        mPoissonsRatio(aPoissonsRatio)
    {
        computeLameConstants();
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~TensileEnergyDensityLocalMeasure()
    {
    }

    /******************************************************************************//**
     * \brief Evaluate tensile energy density local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aDataMap map to stored data
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    void
    operator()(
        const Plato::ScalarMultiVectorT <StateT>  & aStateWS,
        const Plato::ScalarArray3DT     <ConfigT> & aConfigWS,
              Plato::ScalarVectorT      <ResultT> & aResultWS
    ) override
    {
        const Plato::OrdinalType tNumCells = aResultWS.size();

        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;

        Plato::SmallStrain<ElementType> tComputeCauchyStrain;
        Plato::ComputeGradientMatrix<ElementType> tComputeGradientMatrix;
        Plato::Eigenvalues<mNumSpatialDims, mNumVoigtTerms> tComputeEigenvalues;
        Plato::TensileEnergyDensity<mNumSpatialDims> tComputeTensileEnergyDensity;

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        const Plato::Scalar tLameLambda = mLameConstantLambda;
        const Plato::Scalar tLameMu     = mLameConstantMu;

        Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigT tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigT> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainT> tStrain(0.0);
            Plato::Array<ElementType::mNumSpatialDims, StrainT> tPrincipalStrain(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            tComputeGradientMatrix(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
            tComputeCauchyStrain(iCellOrdinal, tStrain, aStateWS, tGradient);
            tComputeEigenvalues(tStrain, tPrincipalStrain, true);
            tComputeTensileEnergyDensity(iCellOrdinal, tPrincipalStrain, tLameLambda, tLameMu, aResultWS);
        });
    }
};
// class TensileEnergyDensityLocalMeasure

}
//namespace Plato

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_DEC_3(Plato::TensileEnergyDensityLocalMeasure, Plato::MechanicsElement)
