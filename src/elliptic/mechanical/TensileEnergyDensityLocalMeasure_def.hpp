#pragma once

#include "TensileEnergyDensityLocalMeasure_decl.hpp"

#include "SmallStrain.hpp"
#include "Eigenvalues.hpp"
#include "LinearStress.hpp"
#include "PlatoMathTypes.hpp"
#include "GradientMatrix.hpp"
#include "TensileEnergyDensity.hpp"

namespace Plato
{
    /******************************************************************************//**
     * \brief Get Youngs Modulus and Poisson's Ratio from input parameter list
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename EvaluationType>
    void
    TensileEnergyDensityLocalMeasure<EvaluationType>::
    getYoungsModulusAndPoissonsRatio(Teuchos::ParameterList & aInputParams)
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
    template<typename EvaluationType>
    void
    TensileEnergyDensityLocalMeasure<EvaluationType>::
    computeLameConstants()
    {
        mLameConstantMu     = mYoungsModulus / 
                             (static_cast<Plato::Scalar>(2.0) * (static_cast<Plato::Scalar>(1.0) + 
                              mPoissonsRatio));
        mLameConstantLambda = static_cast<Plato::Scalar>(2.0) * mLameConstantMu * mPoissonsRatio / 
                             (static_cast<Plato::Scalar>(1.0) - static_cast<Plato::Scalar>(2.0) * mPoissonsRatio);
    }

    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    template<typename EvaluationType>
    TensileEnergyDensityLocalMeasure<EvaluationType>::
    TensileEnergyDensityLocalMeasure(
        const Plato::SpatialDomain   & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) : 
        AbstractLocalMeasure<EvaluationType>(aSpatialModel, aDataMap, aInputParams, aName)
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
    template<typename EvaluationType>
    TensileEnergyDensityLocalMeasure<EvaluationType>::
    TensileEnergyDensityLocalMeasure(
        const Plato::SpatialDomain & aSpatialModel,
              Plato::DataMap       & aDataMap,
        const Plato::Scalar        & aYoungsModulus,
        const Plato::Scalar        & aPoissonsRatio,
        const std::string          & aName
    ) :
        AbstractLocalMeasure<EvaluationType>(aSpatialModel, aDataMap, aName),
        mYoungsModulus(aYoungsModulus),
        mPoissonsRatio(aPoissonsRatio)
    {
        computeLameConstants();
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    template<typename EvaluationType>
    TensileEnergyDensityLocalMeasure<EvaluationType>::
    ~TensileEnergyDensityLocalMeasure()
    {
    }

    /******************************************************************************//**
     * \brief Evaluate tensile energy density local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aDataMap map to stored data
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    template<typename EvaluationType>
    void
    TensileEnergyDensityLocalMeasure<EvaluationType>::
    operator()(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS
    )
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
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
}
//namespace Plato
